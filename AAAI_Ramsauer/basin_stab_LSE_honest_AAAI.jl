#=
GPU-Accelerated HONEST LSE Basin Stability — AAAI 2027 paper (saddle-dominated capacity)
────────────────────────────────────────────────────────────────────────────────
Usage
  julia basin_stab_LSE_honest_AAAI.jl              # resume from last completed α
  julia basin_stab_LSE_honest_AAAI.jl --fresh      # overwrite CSV, start over

Purpose
  Honest Monte Carlo over the full α ∈ [0.20, 0.70] grid with N=25 by default.
  All M = ⌈exp(αN)⌉ patterns generated explicitly; no truncation. Companion to
  basin_stab_LSE_semismart_AAAI.jl (which retains only the saddle-band patterns
  with an analytic-bulk correction).

Differences from basin_stab_LSE_honest_fixedN.jl (NIPS_Resilience version)
  • α grid 0.20:0.01:0.70 (was 0.30:0.05:0.70).
  • T grid 40 points via range(0.025, 0.500, length=40) (was 0.025:0.025:0.500).
  • OOM-aware fallback: per-α memory check; first caps n_disorder, then drops
    N if a single disorder sample still won't fit. The per-α N actually used
    is recorded in the CSV column `N_used`.
  • CSV filename basin_stab_LSE_honest_AAAI_N<targetN>.csv, includes header
    metadata and N_used column.

Run pattern: assumes a working CUDA device, matches CLAUDE.md convention.
────────────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter
using Dates: now

# ──────────────── Precision ────────────────
const USE_FLOAT16 = false
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Targets and budgets (edit here, no CLI knobs) ────────────────
const N_TARGET      = 25                  # preferred spin count; auto-reduced if OOM
const N_MIN         = 10                  # do not reduce below this
const betanet       = F(1.0)              # LSE kernel sharpness
const PHI_MIN       = F(1.0)              # initialise at ξ¹ exactly
const PHI_MAX       = F(1.0)
const N_EQ          = 2^15                # 32768 equilibration steps
const N_SAMP        = 2^13                # 8192 sampling steps
const MAX_N_TRIALS  = 512                 # disorder trials at α_min
const MIN_N_TRIALS  = 32                  # disorder trials at α_max
const MEM_BUDGET_GB = 40.0                # per-α GPU memory cap

# ──────────────── α, T grids ────────────────
const alpha_vec = collect(F(0.20):F(0.01):F(0.70))                  # 51 values
const T_vec     = collect(F(0.005):F(0.01):F(0.485))                # 49 values, step 0.01
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

# Disorder trial count — geometric ramp from MAX to MIN as α grows
function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / max(1, n_alpha - 1)
    trials = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    return max(MIN_N_TRIALS, trials)
end

# Memory cost (bytes) for one disorder sample at a given (N, M)
# Buffers: patterns N·M, target N, three replicas xa,xb,xp 3·N·nT,
#          overlap M·nT, energies 3·nT, phia,phib,qs,phimax 4·nT, β·nT,
#          random buffer nT  ≈  N·M + 3·N·nT + M·nT + 9·nT  (per disorder)
function mem_per_disorder_bytes(N::Int, M::Int)
    elems = Float64(N)*M + 3*Float64(N)*n_T + Float64(M)*n_T + 9*n_T + n_T
    return elems * sizeof(F)
end

# Resolve (N_used, n_disorder) for the α at position idx to fit the memory budget.
# Strategy: keep N_TARGET; cap n_disorder. If even n_dis=1 won't fit, drop N.
# (Takes idx, not α — Float32 equality on the α grid is fragile.)
function resolve_alpha_budget(idx::Int)
    α = alpha_vec[idx]
    N = N_TARGET
    M = round(Int, exp(α * N))
    n_dis_max = n_trials_for_alpha(idx) ÷ 2
    while N >= N_MIN
        per = mem_per_disorder_bytes(N, M)
        if per <= MEM_BUDGET_GB*1e9
            n_dis = min(n_dis_max, floor(Int, MEM_BUDGET_GB*1e9 / per))
            n_dis = max(1, n_dis)
            return (N, n_dis)
        end
        N -= 1
        M  = round(Int, exp(α * N))
    end
    error(@sprintf("α=%.2f: even N=%d (M=%d) exceeds MEM_BUDGET_GB=%.1f — raise budget or skip this α.",
                   α, N_MIN, round(Int, exp(α*N_MIN)), MEM_BUDGET_GB))
end

# Step size σ(T) = 2.4·T/√N
function make_ss_gpu(N::Int)
    Nf = F(N)
    ss_cpu = F.(2.4 .* T_vec ./ sqrt(Nf))
    return CuArray(reshape(ss_cpu, 1, n_T, 1))
end

# ──────────────── LSE energy (batched over disorder samples) ────────────────
function compute_energy_lse!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = -betanet * (Nf - overlap)
    m = maximum(overlap, dims=1)
    @. overlap = exp(overlap - m)
    s = sum(overlap, dims=1)
    E .= vec(@. -(m + log(s)) / betanet)
    return nothing
end

function compute_phi_max_other!(phi_max_out::CuVector{F}, x::CuArray{F,3},
                                 patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                 Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = overlap / Nf
    overlap[1, :, :] .= F(-1e30)
    mx = maximum(overlap, dims=1)
    phi_max_out .+= vec(mx)
    return nothing
end

function mc_step!(x, xp, E, Ep, pat, ov, β, ra, Nf, ss, n_dis_local::Int)
    randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm
    compute_energy_lse!(Ep, xp, pat, ov, Nf)
    rand!(ra)
    acc = @. (ra < exp(-(β * (Ep - E))))
    a3 = reshape(acc, 1, n_T, n_dis_local)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

function initialise_at_target!(x::CuArray{F,3}, target::CuArray{F,3})
    @views for j in 1:n_T
        x[:, j, :] .= target[:, 1, :]
    end
    return nothing
end

# ──────────────── CLI & Resume ────────────────
const FRESH_START = "--fresh" in ARGS
const csv_out     = @sprintf("basin_stab_LSE_honest_AAAI_N%d.csv", N_TARGET)

function read_completed_alphas(csv_file::String)
    !isfile(csv_file) && return Set{String}()
    seen = Set{String}()
    first_data = true
    open(csv_file, "r") do f
        for line in eachline(f)
            isempty(line) && continue
            startswith(line, "#") && continue
            if first_data; first_data = false; continue; end
            push!(seen, String(split(line, ",")[1]))
        end
    end
    return seen
end

function sort_csv!(csv_file::String)
    !isfile(csv_file) && return
    lines = readlines(csv_file)
    isempty(lines) && return
    meta = String[]; header = ""; data = String[]
    first_data = true
    for l in lines
        if startswith(l, "#"); push!(meta, l); continue; end
        if first_data; header = l; first_data = false; continue; end
        isempty(l) && continue
        push!(data, l)
    end
    function sortkey(l)
        parts = split(l, ",")
        return (parse(Float64, parts[1]),
                parse(Float64, parts[2]),
                parse(Int,     parts[4]))   # disorder column is now 4 (after N_used)
    end
    sort!(data, by=sortkey)
    tmp = csv_file * ".tmp"
    open(tmp, "w") do f
        for m in meta; println(f, m); end
        println(f, header)
        for l in data; println(f, l); end
    end
    mv(tmp, csv_file; force=true)
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("="^76)
    println("HONEST LSE Basin Stability — AAAI 2027 (saddle-dominated capacity)")
    @printf("  N_TARGET = %d (fallback floor %d)   β_net = %s   precision = %d-bit\n",
            N_TARGET, N_MIN, betanet, sizeof(F)*8)
    @printf("  α grid: %.2f : %.2f : %.2f  (%d values)\n",
            alpha_vec[1], alpha_vec[2]-alpha_vec[1], alpha_vec[end], n_alpha)
    @printf("  T grid: %.4f : %.4f  (%d points, increasing)\n",
            T_vec[1], T_vec[end], n_T)
    @printf("  MC: %d eq + %d samp   MEM_BUDGET = %.1f GB\n",
            N_EQ, N_SAMP, MEM_BUDGET_GB)
    println("="^76)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Resolve per-α (N_used, n_disorder)
    resolved = Vector{Tuple{Int,Int}}(undef, n_alpha)
    println("Per-α memory plan:")
    for i in 1:n_alpha
        N_i, n_dis_i = resolve_alpha_budget(i)
        resolved[i] = (N_i, n_dis_i)
        M_i = round(Int, exp(alpha_vec[i] * N_i))
        per = mem_per_disorder_bytes(N_i, M_i) * n_dis_i
        @printf("  α=%.2f  N_used=%-3d  M=%-11d  n_dis=%-3d  est=%.2f GB\n",
                alpha_vec[i], N_i, M_i, n_dis_i, per/1e9)
    end
    println()

    # CSV header / resume
    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# generator=basin_stab_LSE_honest_AAAI.jl  N_TARGET=%d  N_MIN=%d  betanet=%s\n",
                    N_TARGET, N_MIN, betanet)
            @printf(f, "# N_EQ=%d  N_SAMP=%d  MEM_BUDGET_GB=%.1f  generated=%s\n",
                    N_EQ, N_SAMP, MEM_BUDGET_GB, string(now()))
            write(f, "alpha,T,N_used,disorder,phi_a,phi_b,q12,phi_max_other\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No existing CSV, starting fresh.")
    end
    completed = FRESH_START ? Set{String}() : read_completed_alphas(csv_out)
    pending = [i for i in 1:n_alpha if !(@sprintf("%.3f", alpha_vec[i]) in completed)]
    if isempty(pending)
        println("All $n_alpha α values already in CSV. Nothing to do.")
        sort_csv!(csv_out); return
    end
    @printf("Pending α: %d/%d\n\n", length(pending), n_alpha)

    t_total_eq = 0.0; t_total_samp = 0.0

    for gi in pending
        α            = alpha_vec[gi]
        N_used, n_dis = resolved[gi]
        Nf           = F(N_used)
        M            = round(Int, exp(α * N_used))
        n_chains     = n_T * n_dis

        println("─"^76)
        @printf("α=%.2f   N=%d   M=%d   n_dis=%d   n_chains=%d\n",
                α, N_used, M, n_dis, n_chains)
        println("─"^76)

        # ── Generate patterns (CPU → GPU) ──
        Random.seed!(42 + 1000*gi)
        print("  Generating patterns ($(M) × N=$(N_used) × n_dis=$(n_dis)) ... ")
        t0 = time()
        p_cpu = randn(F, N_used, M, n_dis)
        for d in 1:n_dis, j in 1:M
            c = @view p_cpu[:, j, d]
            c .*= sqrt(Nf) / norm(c)
        end
        @printf("%.1f s\n", time() - t0)

        print("  Moving to GPU and allocating workspace ... ")
        t0 = time()
        pats_g    = CuArray(p_cpu)
        tgts_g    = CuArray(p_cpu[:, 1:1, :])
        xa_g      = CUDA.zeros(F, N_used, n_T, n_dis)
        xb_g      = CUDA.zeros(F, N_used, n_T, n_dis)
        xp_g      = CUDA.zeros(F, N_used, n_T, n_dis)
        ov_g      = CUDA.zeros(F, M,      n_T, n_dis)
        Ea_g      = CUDA.zeros(F, n_chains)
        Eb_g      = CUDA.zeros(F, n_chains)
        Ep_g      = CUDA.zeros(F, n_chains)
        phia_g    = CUDA.zeros(F, n_chains)
        phib_g    = CUDA.zeros(F, n_chains)
        qs_g      = CUDA.zeros(F, n_chains)
        phimax_g  = CUDA.zeros(F, n_chains)
        β_g       = CuVector{F}(repeat(F.(1 ./ T_vec), n_dis))
        ra_g      = CUDA.zeros(F, n_chains)
        ss_g      = make_ss_gpu(N_used)
        p_cpu = nothing; GC.gc(); CUDA.synchronize()
        @printf("%.1f s\n", time() - t0)

        initialise_at_target!(xa_g, tgts_g)
        initialise_at_target!(xb_g, tgts_g)
        compute_energy_lse!(Ea_g, xa_g, pats_g, ov_g, Nf)
        compute_energy_lse!(Eb_g, xb_g, pats_g, ov_g, Nf)
        CUDA.synchronize()

        # ── Equilibration ──
        println("  Equilibration ($N_EQ steps)…")
        t0 = time()
        prog = Progress(N_EQ, desc="    eq: ")
        for _ in 1:N_EQ
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_eq = time() - t0; t_total_eq += t_eq
        @printf("    %.1f s (%.2f ms/step)\n", t_eq, 1000*t_eq/N_EQ)

        # ── Sampling ──
        phia_g .= zero(F); phib_g .= zero(F)
        qs_g   .= zero(F); phimax_g .= zero(F)
        println("  Sampling ($N_SAMP steps)…")
        t0 = time()
        prog = Progress(N_SAMP, desc="    samp: ")
        for _ in 1:N_SAMP
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            phia_g .+= vec(sum(tgts_g .* xa_g, dims=1)) ./ Nf
            phib_g .+= vec(sum(tgts_g .* xb_g, dims=1)) ./ Nf
            qs_g   .+= vec(sum(xa_g .* xb_g, dims=1)) ./ Nf
            compute_phi_max_other!(phimax_g, xa_g, pats_g, ov_g, Nf)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_samp = time() - t0; t_total_samp += t_samp
        @printf("    %.1f s (%.2f ms/step)\n", t_samp, 1000*t_samp/N_SAMP)

        # ── Write CSV ──
        phia_avg   = Array(phia_g)   ./ N_SAMP
        phib_avg   = Array(phib_g)   ./ N_SAMP
        q_avg      = Array(qs_g)     ./ N_SAMP
        phimax_avg = Array(phimax_g) ./ N_SAMP
        phia_mat   = reshape(phia_avg,   n_T, n_dis)
        phib_mat   = reshape(phib_avg,   n_T, n_dis)
        q_mat      = reshape(q_avg,      n_T, n_dis)
        phimax_mat = reshape(phimax_avg, n_T, n_dis)
        open(csv_out, "a") do f
            for d in 1:n_dis, j in 1:n_T
                @printf(f, "%.3f,%.5f,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                        α, T_vec[j], N_used, d,
                        phia_mat[j, d], phib_mat[j, d],
                        q_mat[j, d], phimax_mat[j, d])
            end
        end
        print("  Sorting CSV… "); sort_csv!(csv_out); println("done.")

        # ── Free GPU ──
        pats_g = nothing; tgts_g = nothing
        xa_g = nothing; xb_g = nothing; xp_g = nothing
        ov_g = nothing; Ea_g = nothing; Eb_g = nothing; Ep_g = nothing
        phia_g = nothing; phib_g = nothing; qs_g = nothing; phimax_g = nothing
        β_g = nothing; ra_g = nothing; ss_g = nothing
        GC.gc(); CUDA.reclaim()
    end

    println("\n" * "="^76)
    @printf("CSV saved: %s\n", csv_out)
    @printf("Total time: equilibration %.1f s, sampling %.1f s\n", t_total_eq, t_total_samp)
    println("="^76)
end

main()

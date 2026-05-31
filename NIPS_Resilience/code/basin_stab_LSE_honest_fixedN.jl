#=
GPU-Accelerated HONEST LSE Basin Stability Test — fixed N — PER-SAMPLE OUTPUT
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSE_honest_fixedN.jl           # resume from last completed α
  julia basin_stab_LSE_honest_fixedN.jl --fresh   # overwrite CSV, start from scratch

Goal (AAAI paper):
  Confirm the exact-density LSE retrieval boundary α_c^E(T) at a *fixed*
  spin count N (default N=25), running honest MC with the full pattern set
  M = exp(αN).  Smart MC (basin_stab_LSE_smart_v18.jl) has structural
  limitations that prevent it from reproducing α_c^E(T); honest MC at
  moderate fixed N is the correct tool — limited only by GPU memory for M.

Differences from basin_stab_LSE_v8m_a1.jl:
  • N is FIXED (=25), not derived from α via a geometric M-ramp.
  • α grid 0.30:0.05:0.70.  M = round(exp(αN)) ∈ [1.8e3, 4.0e7].
  • n_disorder per α is capped adaptively to fit in TARGET_MEM_PER_CHUNK_GB.
  • Initialisation at φ=1 exactly (same as v8m_a1).
  • Output filename:  basin_stab_LSE_honest_N<N>.csv
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter
using Dates: now

# ──────────────── Precision Settings ────────────────
const USE_FLOAT16 = false
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Honest-MC Parameters ────────────────
const N             = 25                  # FIXED spin count
const betanet       = F(1.0)              # LSE kernel inverse variance
const PHI_MIN       = F(1.0)              # initialise at ξ¹ exactly
const PHI_MAX       = F(1.0)
const MAX_N_TRIALS  = 512
const MIN_N_TRIALS  = 32
const N_EQ          = 2^15                # 16384 equilibration steps
const N_SAMP        = 2^13                # 8192 sampling steps

# ──────────────── Sweep grids ────────────────
const alpha_vec     = collect(F(0.30):F(0.05):F(0.70))    # 9 values
const T_vec         = collect(F(0.025):F(0.025):F(0.500)) # 20 values
const n_alpha       = length(alpha_vec)
const n_T           = length(T_vec)

const TARGET_MEM_PER_CHUNK_GB = 40.0

# Disorder trial count — geometric ramp from MAX to MIN as α grows
function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / max(1, n_alpha - 1)
    trials = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    return max(MIN_N_TRIALS, trials)
end
const n_trials_vec   = [n_trials_for_alpha(i) for i in 1:n_alpha]
const n_disorder_vec = [n_trials_vec[i] ÷ 2 for i in 1:n_alpha]

# Step size  σ(T) = 2.4·T/√N  (same as v8m_a1)
function make_ss_gpu()
    ss_cpu = F.(2.4 .* T_vec ./ sqrt(F(N)))
    return CuArray(reshape(ss_cpu, 1, n_T, 1))
end

# ──────────────── LSE energy (batched over disorder samples) ────────────────
function compute_energy_lse!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = -betanet * (Nf - overlap)            # arg_μ = -β_net·N·(1-φ_μ)
    m = maximum(overlap, dims=1)
    @. overlap = exp(overlap - m)
    s = sum(overlap, dims=1)
    E .= vec(@. -(m + log(s)) / betanet)
    return nothing
end

# ──────────────── Max overlap with non-target patterns ────────────────
function compute_phi_max_other!(phi_max_out::CuVector{F}, x::CuArray{F,3},
                                 patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                 Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = overlap / Nf
    overlap[1, :, :] .= F(-1e30)                       # exclude champion
    mx = maximum(overlap, dims=1)
    phi_max_out .+= vec(mx)
    return nothing
end

# ──────────────── MC step ────────────────
function mc_step!(x, xp, E, Ep, pat, ov, β, ra, Nf, ss)
    randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lse!(Ep, xp, pat, ov, Nf)

    rand!(ra)
    acc = @. (ra < exp(-(β * (Ep - E))))
    n_dis = length(β) ÷ n_T
    a3 = reshape(acc, 1, n_T, n_dis)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# Initialise both replicas at x = ξ¹ (PHI_MIN = PHI_MAX = 1 means no random kick)
function initialise_at_target!(x::CuArray{F,3}, target::CuArray{F,3})
    @views for j in 1:n_T
        x[:, j, :] .= target[:, 1, :]
    end
    return nothing
end

# ──────────────── CLI & Resume ────────────────
const FRESH_START = "--fresh" in ARGS
const csv_out     = @sprintf("basin_stab_LSE_honest_N%d.csv", N)

function read_completed_alphas(csv_file::String)
    !isfile(csv_file) && return Set{String}()
    seen = Set{String}()
    first_data = true
    open(csv_file, "r") do f
        for line in eachline(f)
            isempty(line) && continue
            startswith(line, "#") && continue
            if first_data
                first_data = false; continue
            end
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
                parse(Int,     parts[3]))
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

    println("=" ^ 72)
    println("HONEST LSE Basin Stability — fixed N")
    @printf("  N = %d (fixed)   β_net = %s   precision = %d-bit\n",
            N, betanet, sizeof(F)*8)
    println("  α range: $(alpha_vec[1]) – $(alpha_vec[end])   step $(alpha_vec[2]-alpha_vec[1])")
    @printf("  Trials: %d (α_min) → %d (α_max)   disorder = trials/2\n",
            MAX_N_TRIALS, MIN_N_TRIALS)
    println("  Equilibration: $N_EQ steps   Sampling: $N_SAMP steps")
    println("  Initial state: x = ξ¹ exactly")
    println("=" ^ 72)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # M(α) and memory budget
    Nf = F(N)
    Ms = [round(Int, exp(α * N)) for α in alpha_vec]
    @printf("M(α):  %s\n", join([@sprintf("α=%.2f→M=%d", alpha_vec[i], Ms[i])
                                  for i in 1:n_alpha], "  "))

    # Cap n_disorder per α to fit memory
    mem_budget = TARGET_MEM_PER_CHUNK_GB * 1e9
    println()
    println("Per-α memory check (capping n_disorder if needed):")
    for i in 1:n_alpha
        M = Ms[i]; n_dis = n_disorder_vec[i]
        # patterns N*M*n_dis + tgts N*1*n_dis + xa,xb,xp N*nT*n_dis (3×)
        # + ov M*nT*n_dis + 9× n_chains float
        cost = (Float64(N*M*n_dis + N*n_dis + 3*N*n_T*n_dis +
                        M*n_T*n_dis + 9*n_T*n_dis + n_T)) * sizeof(F)
        if cost > mem_budget
            # cap n_disorder
            per_dis = cost/n_dis
            new_n_dis = max(1, floor(Int, mem_budget / per_dis))
            n_disorder_vec[i] = new_n_dis
            n_trials_vec[i]   = 2 * new_n_dis
            cost = new_n_dis * per_dis
            @printf("  α=%.2f  M=%-12d  n_dis capped to %d  (%.2f GB)\n",
                    alpha_vec[i], M, new_n_dis, cost/1e9)
        else
            @printf("  α=%.2f  M=%-12d  n_dis=%d  (%.2f GB)\n",
                    alpha_vec[i], M, n_dis, cost/1e9)
        end
    end
    println()

    @printf("Grid: %d α × %d T,  MC: %d eq + %d samp\n\n",
            n_alpha, n_T, N_EQ, N_SAMP)

    # Initialise / resume CSV
    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# N=%d  betanet=%s  generated=%s\n",
                    N, betanet, string(now()))
            write(f, "alpha,T,disorder,phi_a,phi_b,q12,phi_max_other\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No existing CSV found, starting fresh.")
    end
    completed = FRESH_START ? Set{String}() : read_completed_alphas(csv_out)
    pending_indices = Int[]
    for i in 1:n_alpha
        key = @sprintf("%.3f", alpha_vec[i])
        !(key in completed) && push!(pending_indices, i)
    end
    n_pending = length(pending_indices)
    if n_pending == 0
        println("All $n_alpha α values already in CSV. Nothing to do.")
        sort_csv!(csv_out); return
    end
    @printf("Pending α: %d/%d.   indices: %s\n\n", n_pending, n_alpha,
            join([@sprintf("%.2f", alpha_vec[i]) for i in pending_indices], ", "))

    t_total_eq = 0.0; t_total_samp = 0.0

    # Process one α at a time (heavy ones don't share memory with neighbours)
    for gi in pending_indices
        α      = alpha_vec[gi]
        M      = Ms[gi]
        n_dis  = n_disorder_vec[gi]
        n_chains = n_T * n_dis

        println("─"^72)
        @printf("Processing α=%.2f   M=%d   n_dis=%d   n_chains=%d\n", α, M, n_dis, n_chains)
        println("─"^72)

        # ── Generate patterns (CPU → GPU) ──
        Random.seed!(42 + 1000*gi)
        print("  Generating patterns ($(M) × N=$(N) × n_dis=$(n_dis)) ... ")
        t0 = time()
        p_cpu = randn(F, N, M, n_dis)
        for d in 1:n_dis, j in 1:M
            c = @view p_cpu[:, j, d]
            c .*= sqrt(Nf) / norm(c)
        end
        @printf("%.1f s\n", time() - t0)

        # ── Allocate on GPU ──
        print("  Moving to GPU and allocating workspace ... ")
        t0 = time()
        pats_g    = CuArray(p_cpu)
        tgts_g    = CuArray(p_cpu[:, 1:1, :])
        xa_g      = CUDA.zeros(F, N, n_T, n_dis)
        xb_g      = CUDA.zeros(F, N, n_T, n_dis)
        xp_g      = CUDA.zeros(F, N, n_T, n_dis)
        ov_g      = CUDA.zeros(F, M, n_T, n_dis)
        Ea_g      = CUDA.zeros(F, n_chains)
        Eb_g      = CUDA.zeros(F, n_chains)
        Ep_g      = CUDA.zeros(F, n_chains)
        phia_g    = CUDA.zeros(F, n_chains)
        phib_g    = CUDA.zeros(F, n_chains)
        qs_g      = CUDA.zeros(F, n_chains)
        phimax_g  = CUDA.zeros(F, n_chains)
        β_g       = CuVector{F}(repeat(F.(1 ./ T_vec), n_dis))
        ra_g      = CUDA.zeros(F, n_chains)
        ss_g      = make_ss_gpu()
        p_cpu = nothing; GC.gc()
        CUDA.synchronize()
        @printf("%.1f s\n", time() - t0)

        # ── Initialise both replicas at ξ¹ ──
        initialise_at_target!(xa_g, tgts_g)
        initialise_at_target!(xb_g, tgts_g)
        compute_energy_lse!(Ea_g, xa_g, pats_g, ov_g, Nf)
        compute_energy_lse!(Eb_g, xb_g, pats_g, ov_g, Nf)
        CUDA.synchronize()

        # ── Equilibration ──
        println("  Equilibration ($N_EQ steps)...")
        t0 = time()
        prog = Progress(N_EQ, desc="    eq: ")
        for _ in 1:N_EQ
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_eq = time() - t0; t_total_eq += t_eq
        @printf("    %.1f s (%.2f ms/step)\n", t_eq, 1000*t_eq/N_EQ)

        # ── Sampling ──
        phia_g .= zero(F); phib_g .= zero(F)
        qs_g   .= zero(F); phimax_g .= zero(F)
        println("  Sampling ($N_SAMP steps)...")
        t0 = time()
        prog = Progress(N_SAMP, desc="    samp: ")
        for _ in 1:N_SAMP
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g)
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
                @printf(f, "%.3f,%.5f,%d,%.6f,%.6f,%.6f,%.6f\n",
                        α, T_vec[j], d,
                        phia_mat[j, d], phib_mat[j, d],
                        q_mat[j, d], phimax_mat[j, d])
            end
        end
        print("  Sorting CSV... "); sort_csv!(csv_out); println("done.")

        # ── Free GPU memory ──
        pats_g = nothing; tgts_g = nothing
        xa_g = nothing; xb_g = nothing; xp_g = nothing
        ov_g = nothing; Ea_g = nothing; Eb_g = nothing
        Ep_g = nothing; phia_g = nothing; phib_g = nothing
        qs_g = nothing; phimax_g = nothing
        β_g = nothing; ra_g = nothing; ss_g = nothing
        GC.gc(); CUDA.reclaim()
    end

    println("\n" * "=" ^ 72)
    @printf("CSV saved: %s\n", csv_out)
    @printf("Total time: equilibration %.1f s,  sampling %.1f s\n", t_total_eq, t_total_samp)
    println("=" ^ 72)
end

main()

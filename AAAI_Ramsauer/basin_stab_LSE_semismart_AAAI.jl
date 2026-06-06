#=
GPU-Accelerated SEMISMART LSE Basin Stability — AAAI 2027 paper
────────────────────────────────────────────────────────────────────────────────
Saddle-band retention scheme: of the full M=⌈exp(αN)⌉ patterns, retain only
those with φ_1μ ≥ PHI_KEEP (a const at top of the file). Tail patterns with
φ_1μ < PHI_KEEP are replaced by their analytic expectation C_bulk added as a
constant to the LSE log-sum.

Patterns are quenched: sampled once per disorder seed, identity preserved.
This is explicitly different from v21 sticky-cache which minted fresh patterns
whenever the chain visited a new direction (broke quenching, destroyed retrieval).

Usage
  julia basin_stab_LSE_semismart_AAAI.jl              # resume
  julia basin_stab_LSE_semismart_AAAI.jl --fresh      # overwrite
────────────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter
using Dates: now
using QuadGK
using SpecialFunctions

# ──────────────── Precision ────────────────
const USE_FLOAT16 = false
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Targets and budgets ────────────────
const N_TARGET      = 25
const N_MIN         = 10
const betanet       = F(1.0)
const PHI_KEEP      = F(0.40)             # retain φ_1μ ≥ PHI_KEEP; below → analytic bulk
const PHI_MIN       = F(1.0)
const PHI_MAX       = F(1.0)
const N_EQ          = 2^15
const N_SAMP        = 2^13
const MAX_N_TRIALS  = 512
const MIN_N_TRIALS  = 32
const MEM_BUDGET_GB = 40.0

# Hard cap on retained patterns per disorder sample
# (sanity bound; expected K is tiny relative to M for PHI_KEEP=0.40 at our α range)
const K_HARD_CAP    = 5_000_000

# ──────────────── α, T grids ────────────────
const alpha_vec = collect(F(0.20):F(0.01):F(0.70))
const T_vec     = collect(F(0.005):F(0.01):F(0.485))
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

# ──────────────── Sphere-density utilities ────────────────
# f(φ) = (1-φ²)^((N-3)/2) / B((N-1)/2, 1/2), φ ∈ [-1, 1]
# Probability that φ_1μ ≥ PHI_KEEP for a random pattern on the N-sphere
function prob_phi_above(N::Int, phi_lo::Float64)
    a = (N - 3) / 2
    log_norm = -logbeta((N - 1)/2, 0.5)
    integrand(u) = exp(a*log(1 - u^2) + log_norm)
    val, _ = quadgk(integrand, phi_lo, 1.0 - 1e-12; rtol=1e-10)
    return val
end

# log E_φ[ exp(β_net·N·φ) · 1{φ < PHI_KEEP} ]  (over sphere density)
function log_C_bulk_per(N::Int, phi_keep::Float64, bnet::Float64)
    a = (N - 3) / 2
    # locate saddle of f(u) = a·log(1-u²) + β·N·u on (-1, phi_keep) if it lies inside
    A = bnet * N; B = N - 3
    u_sad = (-B + sqrt(B^2 + 4*A^2)) / (2*A)        # in (0, 1)
    f_at(u) = a*log(1 - u^2) + bnet*N*u
    u_ref = min(u_sad, phi_keep - 1e-6)
    f_ref = f_at(u_ref)
    integrand(u) = exp(a*log(1 - u^2) + bnet*N*u - f_ref)
    val, _ = quadgk(integrand, -1.0 + 1e-12, phi_keep; rtol=1e-10)
    log_norm = -logbeta((N - 1)/2, 0.5)
    return f_ref + log(val) + log_norm
end

# Generate retained patterns by rejection sampling on φ_1μ ≥ PHI_KEEP.
# Returns CPU array p_cpu[1:N, 1:K, 1:n_dis] with K = max retained count
# across disorders (padded with random tail samples to keep rectangular,
# but masked via K_per_dis vector at energy time). Simpler: take K = upper
# percentile so all disorders have at least K_min retained, drop excess.
function generate_semismart_patterns(N::Int, M::Int, n_dis::Int, phi_keep::Float64)
    # Expected K per disorder = M * P(φ ≥ phi_keep)
    p_above = prob_phi_above(N, phi_keep)
    K_exp = max(1, ceil(Int, M * p_above))
    # Allocate generous K (4× expectation, capped) for variance headroom
    K_alloc = min(K_HARD_CAP, ceil(Int, 4 * K_exp + 16))
    p_cpu = zeros(F, N, K_alloc, n_dis)
    K_per_dis = zeros(Int, n_dis)
    Nf = F(N)
    sqrtN = sqrt(Nf)
    for d in 1:n_dis
        target = randn(F, N); target .*= sqrtN / norm(target)
        K_kept = 0
        attempts = 0
        max_attempts = M
        while K_kept < K_alloc && attempts < max_attempts
            attempts += 1
            cand = randn(F, N); cand .*= sqrtN / norm(cand)
            φ = sum(cand .* target) / Nf
            if φ ≥ phi_keep
                K_kept += 1
                p_cpu[:, K_kept, d] .= cand
            end
        end
        # Always include the target as pattern μ=1 (slot 1 reserved)
        # Shift retained patterns by +1 and insert target at slot 1
        if K_kept ≥ K_alloc
            K_kept = K_alloc - 1
        end
        for j in K_kept:-1:1
            p_cpu[:, j+1, d] .= @view p_cpu[:, j, d]
        end
        p_cpu[:, 1, d] .= target
        K_per_dis[d] = K_kept + 1
    end
    K_used = maximum(K_per_dis)
    return p_cpu[:, 1:K_used, :], K_per_dis
end

function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / max(1, n_alpha - 1)
    trials = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    return max(MIN_N_TRIALS, trials)
end

function mem_per_disorder_bytes(N::Int, K::Int)
    elems = Float64(N)*K + 3*Float64(N)*n_T + Float64(K)*n_T + 9*n_T + n_T
    return elems * sizeof(F)
end

function resolve_alpha_budget(idx::Int)
    α = alpha_vec[idx]
    N = N_TARGET
    while N >= N_MIN
        M = round(Int, exp(α * N))
        p_above = prob_phi_above(N, Float64(PHI_KEEP))
        K_est = max(8, ceil(Int, 4 * M * p_above + 16))
        K_est = min(K_HARD_CAP, K_est)
        per = mem_per_disorder_bytes(N, K_est)
        if per <= MEM_BUDGET_GB*1e9
            n_dis_max = n_trials_for_alpha(idx) ÷ 2
            n_dis = min(n_dis_max, floor(Int, MEM_BUDGET_GB*1e9 / per))
            n_dis = max(1, n_dis)
            return (N, n_dis, K_est)
        end
        N -= 1
    end
    error(@sprintf("α=%.2f: even N=%d won't fit MEM_BUDGET_GB=%.1f", α, N_MIN, MEM_BUDGET_GB))
end

function make_ss_gpu(N::Int)
    Nf = F(N)
    ss_cpu = F.(2.4 .* T_vec ./ sqrt(Nf))
    return CuArray(reshape(ss_cpu, 1, n_T, 1))
end

# ──────────────── Semismart energy ────────────────
# Standard log-sum-exp over retained K patterns, plus analytic bulk constant
# added inside the log:
#   H = -1/β_net · log(Σ_{retained} exp(β_net·ξ·x) + C_bulk)
# where C_bulk = exp(α·N + log_C_bulk_per) on the same exp-of-Nφ scale.
function compute_energy_semismart!(E::CuVector{F}, x::CuArray{F,3},
                                    patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                    Nf::F, log_C_bulk_total::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = betanet * overlap                         # = β_net · ξ_μ · x
    m          = maximum(overlap, dims=1)
    m_eff      = max.(m, log_C_bulk_total)
    s_retained = sum(exp.(overlap .- m_eff), dims=1)
    s_bulk     = exp.(log_C_bulk_total .- m_eff)
    E         .= vec(@. -(m_eff + log(s_retained + s_bulk)) / betanet)
    return nothing
end

function compute_phi_max_retained!(phi_max_out::CuVector{F}, x::CuArray{F,3},
                                    patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                    Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = overlap / Nf
    overlap[1, :, :] .= F(-1e30)
    mx = maximum(overlap, dims=1)
    phi_max_out .+= vec(mx)
    return nothing
end

function mc_step!(x, xp, E, Ep, pat, ov, β, ra, Nf, ss, n_dis_local::Int, log_C_bulk_total::F)
    randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm
    compute_energy_semismart!(Ep, xp, pat, ov, Nf, log_C_bulk_total)
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
const csv_out     = @sprintf("basin_stab_LSE_semismart_AAAI_N%d_phikeep%.2f.csv",
                              N_TARGET, Float64(PHI_KEEP))

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
                parse(Int,     parts[5]))   # disorder column at position 5
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
    println("SEMISMART LSE Basin Stability — AAAI 2027")
    @printf("  N_TARGET = %d  N_MIN = %d   β_net = %s   φ_keep = %.3f\n",
            N_TARGET, N_MIN, betanet, Float64(PHI_KEEP))
    @printf("  α grid: %.2f : %.2f : %.2f  (%d values)\n",
            alpha_vec[1], alpha_vec[2]-alpha_vec[1], alpha_vec[end], n_alpha)
    @printf("  T grid: %.4f : %.4f  (%d points, increasing)\n",
            T_vec[1], T_vec[end], n_T)
    @printf("  MC: %d eq + %d samp   MEM_BUDGET = %.1f GB\n",
            N_EQ, N_SAMP, MEM_BUDGET_GB)
    println("="^76)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    resolved = Vector{Tuple{Int,Int,Int}}(undef, n_alpha)  # (N_used, n_dis, K_alloc)
    println("Per-α memory plan:")
    for i in 1:n_alpha
        Ni, ni, Ki = resolve_alpha_budget(i)
        resolved[i] = (Ni, ni, Ki)
        per = mem_per_disorder_bytes(Ni, Ki) * ni
        Mi = round(Int, exp(alpha_vec[i] * Ni))
        @printf("  α=%.2f  N_used=%-3d  M=%-12d  K_alloc=%-6d  n_dis=%-3d  est=%.2f GB\n",
                alpha_vec[i], Ni, Mi, Ki, ni, per/1e9)
    end
    println()

    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# generator=basin_stab_LSE_semismart_AAAI.jl  N_TARGET=%d  N_MIN=%d  betanet=%s  phi_keep=%.3f\n",
                    N_TARGET, N_MIN, betanet, Float64(PHI_KEEP))
            @printf(f, "# N_EQ=%d  N_SAMP=%d  MEM_BUDGET_GB=%.1f  generated=%s\n",
                    N_EQ, N_SAMP, MEM_BUDGET_GB, string(now()))
            write(f, "alpha,T,N_used,K_retained,disorder,phi_a,phi_b,q12,phi_max_retained\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No existing CSV, starting fresh.")
    end
    completed = FRESH_START ? Set{String}() : read_completed_alphas(csv_out)
    pending = [i for i in 1:n_alpha if !(@sprintf("%.3f", alpha_vec[i]) in completed)]
    if isempty(pending)
        println("All $n_alpha α values already in CSV."); sort_csv!(csv_out); return
    end
    @printf("Pending α: %d/%d\n\n", length(pending), n_alpha)

    t_total_eq = 0.0; t_total_samp = 0.0

    for gi in pending
        α            = alpha_vec[gi]
        N_used, n_dis, _ = resolved[gi]
        Nf           = F(N_used)
        M_full       = round(Int, exp(α * N_used))

        # Analytic bulk constant for the tail φ < PHI_KEEP
        log_C_bulk_per_val = log_C_bulk_per(Int(N_used), Float64(PHI_KEEP), Float64(betanet))
        log_C_bulk_total   = F(α*N_used + log_C_bulk_per_val)

        println("─"^76)
        @printf("α=%.2f   N=%d   M_full=%d   log_C_bulk_total=%.3f\n",
                α, N_used, M_full, Float64(log_C_bulk_total))

        Random.seed!(42 + 1000*gi)
        print("  Sampling retained patterns (φ_1μ ≥ $(Float64(PHI_KEEP))) ... ")
        t0 = time()
        p_cpu, K_per_dis = generate_semismart_patterns(N_used, M_full, n_dis,
                                                       Float64(PHI_KEEP))
        K_used = size(p_cpu, 2)
        @printf("K_max=%d  K_min=%d  (avg=%d)   %.1f s\n",
                maximum(K_per_dis), minimum(K_per_dis),
                round(Int, mean(K_per_dis)), time()-t0)

        n_chains = n_T * n_dis
        print("  Moving to GPU and allocating workspace ... ")
        t0 = time()
        pats_g    = CuArray(p_cpu)
        tgts_g    = CuArray(p_cpu[:, 1:1, :])
        xa_g      = CUDA.zeros(F, N_used, n_T, n_dis)
        xb_g      = CUDA.zeros(F, N_used, n_T, n_dis)
        xp_g      = CUDA.zeros(F, N_used, n_T, n_dis)
        ov_g      = CUDA.zeros(F, K_used, n_T, n_dis)
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
        compute_energy_semismart!(Ea_g, xa_g, pats_g, ov_g, Nf, log_C_bulk_total)
        compute_energy_semismart!(Eb_g, xb_g, pats_g, ov_g, Nf, log_C_bulk_total)
        CUDA.synchronize()

        println("  Equilibration ($N_EQ steps)…")
        t0 = time(); prog = Progress(N_EQ, desc="    eq: ")
        for _ in 1:N_EQ
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_eq = time() - t0; t_total_eq += t_eq
        @printf("    %.1f s (%.2f ms/step)\n", t_eq, 1000*t_eq/N_EQ)

        phia_g .= zero(F); phib_g .= zero(F); qs_g .= zero(F); phimax_g .= zero(F)
        println("  Sampling ($N_SAMP steps)…")
        t0 = time(); prog = Progress(N_SAMP, desc="    samp: ")
        for _ in 1:N_SAMP
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            phia_g .+= vec(sum(tgts_g .* xa_g, dims=1)) ./ Nf
            phib_g .+= vec(sum(tgts_g .* xb_g, dims=1)) ./ Nf
            qs_g   .+= vec(sum(xa_g .* xb_g, dims=1)) ./ Nf
            compute_phi_max_retained!(phimax_g, xa_g, pats_g, ov_g, Nf)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_samp = time() - t0; t_total_samp += t_samp
        @printf("    %.1f s (%.2f ms/step)\n", t_samp, 1000*t_samp/N_SAMP)

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
                @printf(f, "%.3f,%.5f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                        α, T_vec[j], N_used, K_per_dis[d], d,
                        phia_mat[j, d], phib_mat[j, d],
                        q_mat[j, d], phimax_mat[j, d])
            end
        end
        print("  Sorting CSV… "); sort_csv!(csv_out); println("done.")

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

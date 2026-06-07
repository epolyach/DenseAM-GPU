#=
GPU-Accelerated SEMISMART LSE Basin Stability — AAAI 2027  ·  N-SWEEP
────────────────────────────────────────────────────────────────────────────────
FSS variant of basin_stab_LSE_semismart_AAAI.jl. Loops N over N_LIST (large →
small) and writes one CSV per N. Purpose: show that as N grows the basin
boundary in the (α, T) plane converges to the analytical red curve
α_c^sd(T) = -G_max - f_ret(T)   with   G_max = ½ log φ* - φ*²  (golden φ*).

Key changes vs the base semismart code:
  1. N is the *outer* loop variable; PHI_KEEP is auto-tuned per (N, α) to keep
     K_retained ≈ K_TARGET. This prevents the rejection sampler from blowing
     up at large N (where M = exp(αN) is astronomical).
  2. Direct conditional sampling on the truncated Beta((N-1)/2,(N-1)/2)
     density for (1+φ)/2. Rejection sampling over all M is impossible for
     N ≳ 50 at high α; the inverse-CDF approach below scales to any N.
  3. (α, N) cells where the required PHI_KEEP exceeds PHI_KEEP_MAX are
     skipped and logged. At large N the small-α end is well-covered and
     high-α gets dropped — exactly the regime we care about for FSS toward
     the red curve which the data undershoots most at small α.

Same energy/MC step as basin_stab_LSE_semismart_AAAI.jl. CSV schema matches
so plot_LSE_AAAI_heatmap.jl reads each N's CSV without modification.

Usage
  julia basin_stab_LSE_semismart_AAAI_Nsweep.jl            # resume
  julia basin_stab_LSE_semismart_AAAI_Nsweep.jl --fresh    # overwrite

Output (one per N): basin_stab_LSE_semismart_AAAI_Nsweep_N<N>.csv
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
using Distributions

# ──────────────── Precision ────────────────
const F = Float32

# ──────────────── N sweep + budget ────────────────
const N_LIST       = [100, 75, 50, 25]    # large → small
const K_TARGET     = 30_000               # target retained-pattern count per disorder
const K_HARD_CAP   = 200_000
const PHI_KEEP_MIN = 0.10                 # never go below this — bulk integral assumes a cut
const PHI_KEEP_MAX = 0.97                 # if a higher cut is needed → (N, α) infeasible

const betanet       = F(1.0)
const N_EQ          = 2^15
const N_SAMP        = 2^13
const MAX_N_TRIALS  = 256                 # per-α disorder count cap (high-N is expensive)
const MIN_N_TRIALS  = 16
const MEM_BUDGET_GB = 40.0

# ──────────────── α, T grids ────────────────
const alpha_vec = collect(F(0.20):F(0.01):F(0.70))
const T_vec     = collect(F(0.005):F(0.01):F(0.485))
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

# ──────────────── Sphere-density utilities ────────────────
# Density of φ = ξ·x/|ξ||x| for uniform ξ on S^{N-1} (with x fixed): f(φ) ∝ (1-φ²)^((N-3)/2).
# Equivalently Y = (1+φ)/2 ~ Beta((N-1)/2, (N-1)/2).
function prob_phi_above(N::Int, phi_lo::Float64)
    # P(φ ≥ phi_lo) under the sphere density.
    phi_lo <= -1.0 && return 1.0
    phi_lo >= 1.0 && return 0.0
    a = (N - 1) / 2
    y_lo = (1 + phi_lo) / 2
    _, q = beta_inc(a, a, y_lo, 1 - y_lo)
    return q
end

# ∫_{-1}^{phi_keep} f(φ) exp(β·N·φ) dφ — analytic bulk added inside the LSE log.
function log_C_bulk_per(N::Int, phi_keep::Float64, bnet::Float64)
    a = (N - 3) / 2
    A = bnet * N; B = N - 3
    u_sad = (-B + sqrt(B^2 + 4*A^2)) / (2*A)
    f_at(u) = a*log(1 - u^2) + bnet*N*u
    u_ref = min(u_sad, phi_keep - 1e-6)
    f_ref = f_at(u_ref)
    integrand(u) = exp(a*log(1 - u^2) + bnet*N*u - f_ref)
    val, _ = quadgk(integrand, -1.0 + 1e-12, phi_keep; rtol=1e-10)
    log_norm = -logbeta((N - 1)/2, 0.5)
    return f_ref + log(val) + log_norm
end

# Choose PHI_KEEP per (N, α) so that M·P(φ ≥ PHI_KEEP) ≈ K_TARGET.
# Returns (phi_keep, K_alloc, status) where status ∈ (:honest, :smart, :infeasible).
function pick_phi_keep(N::Int, α::Float64)
    M_full = round(Int, exp(α * N))
    # If full pattern set already fits the budget, do honest MC (no truncation).
    if M_full <= K_TARGET
        K_alloc = min(K_HARD_CAP, M_full)
        return (-1.0, K_alloc, :honest, M_full, 1.0)
    end
    # Otherwise binary-search φ on [PHI_KEEP_MIN, PHI_KEEP_MAX] for M·P(φ≥c) ≈ K_TARGET.
    lo, hi = PHI_KEEP_MIN, PHI_KEEP_MAX
    Klo = Float64(M_full) * prob_phi_above(N, lo)
    Khi = Float64(M_full) * prob_phi_above(N, hi)
    # If even at PHI_KEEP_MAX K is too large → infeasible (would blow memory / lose physics).
    if Khi > K_TARGET
        return (PHI_KEEP_MAX, ceil(Int, Khi), :infeasible, M_full, Khi/M_full)
    end
    # If even at PHI_KEEP_MIN K is below target, no need to go lower (would just add bulk).
    if Klo <= K_TARGET
        # PHI_KEEP_MIN already gives K ≤ target; use it (less aggressive cut, smaller bias).
        p_above = prob_phi_above(N, lo)
        return (lo, max(8, ceil(Int, Klo * 1.5 + 16)), :smart, M_full, p_above)
    end
    # Standard binary search.
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        Kmid = Float64(M_full) * prob_phi_above(N, mid)
        if Kmid > K_TARGET
            lo = mid
        else
            hi = mid
        end
        (hi - lo) < 1e-4 && break
    end
    phi_keep = 0.5 * (lo + hi)
    p_above  = prob_phi_above(N, phi_keep)
    K_alloc  = min(K_HARD_CAP, max(8, ceil(Int, M_full * p_above * 1.5 + 16)))
    return (phi_keep, K_alloc, :smart, M_full, p_above)
end

# ──────────────── Direct conditional pattern sampling ────────────────
# For each disorder seed: draw ξ¹ uniformly on the √N-sphere, then sample
# K_kept patterns ξᵐ with φ_1ᵐ ≥ phi_keep using inverse-CDF on the
# truncated Beta((N-1)/2, (N-1)/2) density (no rejection over M).
function sample_patterns!(p_cpu::Array{F,3}, K_per_dis::Vector{Int},
                          N::Int, n_dis::Int, phi_keep::Float64,
                          K_kept_target::Int, M_full::Int)
    Nf = F(N); sqrtN = sqrt(Nf)
    if phi_keep < 0
        # Honest mode: M_full ≤ K_TARGET; emit all M_full patterns per disorder.
        K_kept_target = min(M_full, K_kept_target)
        @inbounds for d in 1:n_dis
            target = randn(F, N); target .*= sqrtN / norm(target)
            p_cpu[:, 1, d] .= target
            for k in 1:K_kept_target-1
                cand = randn(F, N); cand .*= sqrtN / norm(cand)
                p_cpu[:, k+1, d] .= cand
            end
            K_per_dis[d] = K_kept_target
        end
        return
    end
    a = (N - 1) / 2
    base = Beta(a, a)
    y_lo = (1 + phi_keep) / 2
    td = truncated(base, y_lo, 1.0)
    @inbounds for d in 1:n_dis
        target = randn(F, N); target .*= sqrtN / norm(target)
        p_cpu[:, 1, d] .= target
        # K_kept = number of patterns to draw above the cut for this disorder.
        # Use Poisson around the expected count to capture mean+variance honestly.
        K_kept = min(K_kept_target, max(1, rand(Poisson(M_full * (1 - cdf(base, y_lo))))))
        for k in 1:K_kept
            φ = F(2 * rand(td) - 1)
            perp = randn(F, N)
            proj = sum(perp .* target) / Nf
            @. perp = perp - proj * target
            perp_norm = norm(perp)
            scale = F(sqrt(Float64(Nf) * (1.0 - Float64(φ)^2))) / max(perp_norm, F(1e-12))
            @. perp = perp * scale
            @views @. p_cpu[:, k+1, d] = φ * target + perp
        end
        K_per_dis[d] = K_kept + 1
    end
end

# ──────────────── Memory budgeting ────────────────
function mem_per_disorder_bytes(N::Int, K::Int)
    elems = Float64(N)*K + 3*Float64(N)*n_T + Float64(K)*n_T + 9*n_T + n_T
    return elems * sizeof(F)
end

function pick_n_dis(N::Int, K::Int, alpha_idx::Int)
    t = (alpha_idx - 1) / max(1, n_alpha - 1)
    cap = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    cap = max(MIN_N_TRIALS, cap) ÷ 2
    per = mem_per_disorder_bytes(N, K)
    by_mem = floor(Int, MEM_BUDGET_GB * 1e9 / per)
    return clamp(min(cap, by_mem), 1, cap)
end

# ──────────────── Energy (same as semismart_AAAI) ────────────────
function compute_energy_semismart!(E::CuVector{F}, x::CuArray{F,3},
                                    patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                    Nf::F, log_C_bulk_total::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = betanet * overlap
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

function make_ss_gpu(N::Int)
    Nf = F(N)
    ss_cpu = F.(2.4 .* T_vec ./ sqrt(Nf))
    return CuArray(reshape(ss_cpu, 1, n_T, 1))
end

function initialise_at_target!(x::CuArray{F,3}, target::CuArray{F,3})
    @views for j in 1:n_T
        x[:, j, :] .= target[:, 1, :]
    end
    return nothing
end

# ──────────────── CLI & Resume ────────────────
const FRESH_START = "--fresh" in ARGS

csv_name(N::Int) = @sprintf("basin_stab_LSE_semismart_AAAI_Nsweep_N%d.csv", N)

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
                parse(Int,     parts[5]))   # disorder index column
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

# ──────────────── Per-N driver ────────────────
function run_for_N(N::Int)
    println("\n" * "█"^76)
    @printf("Running N = %d\n", N)
    println("█"^76)
    csv_out = csv_name(N)

    # Plan: per-α resolve PHI_KEEP, K_alloc, n_dis.
    plan = Vector{NamedTuple}(undef, n_alpha)
    println("Per-α plan (φ_keep, K_alloc, n_dis, M_full):")
    for i in 1:n_alpha
        α = Float64(alpha_vec[i])
        phi_keep, K_alloc, status, M_full, p_above = pick_phi_keep(N, α)
        n_dis = status == :infeasible ? 0 : pick_n_dis(N, K_alloc, i)
        plan[i] = (alpha_idx=i, phi_keep=phi_keep, K_alloc=K_alloc,
                   status=status, M_full=M_full, p_above=p_above, n_dis=n_dis)
        @printf("  α=%.2f  status=%-11s  φ_keep=%+5.3f  K_alloc=%-7d  M_full=%-12d  n_dis=%-3d\n",
                α, String(status), phi_keep, K_alloc, M_full, n_dis)
    end

    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# generator=basin_stab_LSE_semismart_AAAI_Nsweep.jl  N=%d  K_TARGET=%d  betanet=%s\n",
                    N, K_TARGET, betanet)
            @printf(f, "# N_EQ=%d  N_SAMP=%d  MEM_BUDGET_GB=%.1f  generated=%s\n",
                    N_EQ, N_SAMP, MEM_BUDGET_GB, string(now()))
            write(f, "alpha,T,N_used,K_retained,disorder,phi_a,phi_b,q12,phi_max_retained\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No CSV — starting fresh.")
    end
    completed = FRESH_START ? Set{String}() : read_completed_alphas(csv_out)
    pending = [p for p in plan if p.status != :infeasible &&
                                  !(@sprintf("%.3f", alpha_vec[p.alpha_idx]) in completed)]
    if isempty(pending)
        println("All feasible α already in CSV.")
        sort_csv!(csv_out); return
    end
    @printf("Pending α: %d   (skipping %d infeasible)\n\n",
            length(pending), count(p -> p.status == :infeasible, plan))

    Nf = F(N)
    for p in pending
        α            = Float64(alpha_vec[p.alpha_idx])
        phi_keep     = p.phi_keep
        K_alloc      = p.K_alloc
        n_dis        = p.n_dis
        M_full       = p.M_full

        # Analytic bulk constant: zero if honest (no truncation).
        log_C_bulk_total = if phi_keep < 0
            F(-1e30)
        else
            F(α*N + log_C_bulk_per(N, phi_keep, Float64(betanet)))
        end

        println("─"^76)
        @printf("α=%.3f   N=%d   M_full=%d   φ_keep=%+.3f   K_alloc=%d   n_dis=%d   log_C_bulk=%.3f\n",
                α, N, M_full, phi_keep, K_alloc, n_dis, Float64(log_C_bulk_total))

        Random.seed!(42 + 1000*p.alpha_idx + 10*N)
        print("  Sampling retained patterns ... ")
        t0 = time()
        p_cpu = zeros(F, N, K_alloc, n_dis)
        K_per_dis = zeros(Int, n_dis)
        sample_patterns!(p_cpu, K_per_dis, N, n_dis, phi_keep, K_alloc, M_full)
        K_used = maximum(K_per_dis)
        p_cpu = p_cpu[:, 1:K_used, :]
        @printf("K_max=%d  K_min=%d  (avg=%d)   %.1f s\n",
                maximum(K_per_dis), minimum(K_per_dis),
                round(Int, mean(K_per_dis)), time()-t0)

        n_chains = n_T * n_dis
        print("  Allocating GPU workspace ... ")
        t0 = time()
        pats_g    = CuArray(p_cpu)
        tgts_g    = CuArray(p_cpu[:, 1:1, :])
        xa_g      = CUDA.zeros(F, N, n_T, n_dis)
        xb_g      = CUDA.zeros(F, N, n_T, n_dis)
        xp_g      = CUDA.zeros(F, N, n_T, n_dis)
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
        ss_g      = make_ss_gpu(N)
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
        @printf("    %.1f s\n", time()-t0)

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
        @printf("    %.1f s\n", time()-t0)

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
                        α, T_vec[j], N, K_per_dis[d], d,
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
    @printf("\nN=%d done.  CSV: %s\n", N, csv_out)
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("="^76)
    println("SEMISMART LSE Basin Stability — AAAI 2027 N-SWEEP")
    @printf("  N_LIST = %s   K_TARGET = %d\n", N_LIST, K_TARGET)
    @printf("  α grid: %.2f : %.2f : %.2f  (%d values)\n",
            alpha_vec[1], alpha_vec[2]-alpha_vec[1], alpha_vec[end], n_alpha)
    @printf("  T grid: %.4f : %.4f  (%d points)\n", T_vec[1], T_vec[end], n_T)
    @printf("  MC: %d eq + %d samp   MEM_BUDGET = %.1f GB\n",
            N_EQ, N_SAMP, MEM_BUDGET_GB)
    println("="^76)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    for N in N_LIST
        try
            run_for_N(N)
        catch err
            @warn "N=$N failed" exception=(err, catch_backtrace())
        end
    end

    println("\n" * "="^76)
    println("All N values complete.")
    println("="^76)
end

main()

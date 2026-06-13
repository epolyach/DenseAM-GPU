#=
GPU-Accelerated SEMISMART LSE Basin Stability — AAAI 2026  ·  N(α)-RAMP, PHI_KEEP fixed
────────────────────────────────────────────────────────────────────────────────
Original semismart design: PHI_KEEP and M are inputs, N(α) is the load-anchored
ramp, K is derived per (α, N) as K = M · P(φ ≥ PHI_KEEP).

Inputs (constants below):
  • M_TARGET   = 4.4e6
  • PHI_KEEP   = 0.40
  • N(α)       = max(N_FLOOR, round(log(M_TARGET) / α))   (matches honest Nramp)
  • All MC and grid params match basin_stab_LSE_honest_AAAI_Nramp.jl
    (N_EQ, N_SAMP, N_DIS_TARGET, MEM_BUDGET_GB, MEM_SAFETY, α grid, T grid).

Derived per (α, N):
  • p_above = P(φ ≥ PHI_KEEP)   under spherical Beta((N-1)/2, (N-1)/2)
  • K_expected = M_full · p_above
  • K_alloc    = clamp(ceil(K_expected · 1.5 + 16), 8, K_HARD_CAP)
  • bulk constant = α·N + log ∫_{-1}^{PHI_KEEP} f(φ) e^{N β φ} dφ

Same energy / MC step as basin_stab_LSE_semismart_AAAI.jl. CSV schema matches
the existing semismart files; plot_LSE_AAAI_heatmap.jl reads it unchanged.

Usage
  julia basin_stab_LSE_semismart_AAAI_Nramp_phikeep.jl            # resume
  julia basin_stab_LSE_semismart_AAAI_Nramp_phikeep.jl --fresh    # overwrite

Output: basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6_phikeep0.40.csv
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

const F = Float32

# ──────────────── Inputs: PHI_KEEP and M; N(α) and K are derived ────────────────
const M_TARGET     = 4.4e6
const PHI_KEEP     = 0.40
const N_FLOOR      = 12
const K_HARD_CAP   = 500_000          # safety cap on K_alloc

const betanet       = F(1.0)
const N_EQ          = 2^15            # match honest Nramp
const N_SAMP        = 2^13            # match honest Nramp
const N_DIS_TARGET  = 32              # match honest Nramp
const MEM_BUDGET_GB = 45.0            # match honest Nramp
const MEM_SAFETY    = 0.62            # match honest Nramp (≈ 27.9 GB usable)

const alpha_vec = collect(F(0.20):F(0.01):F(0.70))
const T_vec     = collect(F(0.005):F(0.01):F(0.495))
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

N_for_alpha(α::Real) = max(N_FLOOR, round(Int, log(M_TARGET) / Float64(α)))

# ──────────────── Sphere-density helpers ────────────────
function prob_phi_above(N::Int, phi_lo::Float64)
    phi_lo <= -1.0 && return 1.0
    phi_lo >=  1.0 && return 0.0
    a = (N - 1) / 2
    y_lo = (1 + phi_lo) / 2
    _, q = beta_inc(a, a, y_lo, 1 - y_lo)
    return q
end

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

# ──────────────── Derive K from inputs (PHI_KEEP fixed, K computed) ────────────────
# Returns (phi_keep, K_alloc, status, M_full, p_above).
# :honest      → M_full ≤ K_HARD_CAP, keep all patterns explicitly, no truncation.
# :smart       → standard truncation at PHI_KEEP, K_alloc derived from M·p_above.
# :infeasible  → derived K_alloc would exceed K_HARD_CAP (skip α).
function derive_K(N::Int, α::Float64)
    M_full = round(Int, exp(α * N))
    if M_full <= K_HARD_CAP
        return (-1.0, min(K_HARD_CAP, M_full), :honest, M_full, 1.0)
    end
    p_above = prob_phi_above(N, PHI_KEEP)
    K_expected = Float64(M_full) * p_above
    K_alloc = max(8, ceil(Int, K_expected * 1.5 + 16))
    if K_alloc > K_HARD_CAP
        return (PHI_KEEP, K_HARD_CAP, :infeasible, M_full, p_above)
    end
    return (PHI_KEEP, K_alloc, :smart, M_full, p_above)
end

# ──────────────── Direct conditional pattern sampling ────────────────
function sample_patterns!(p_cpu::Array{F,3}, K_per_dis::Vector{Int},
                          N::Int, n_dis::Int, phi_keep::Float64,
                          K_alloc::Int, M_full::Int)
    Nf = F(N); sqrtN = sqrt(Nf)
    if phi_keep < 0
        K_emit = min(M_full, K_alloc)
        @inbounds for d in 1:n_dis
            target = randn(F, N); target .*= sqrtN / norm(target)
            p_cpu[:, 1, d] .= target
            for k in 1:K_emit-1
                cand = randn(F, N); cand .*= sqrtN / norm(cand)
                p_cpu[:, k+1, d] .= cand
            end
            K_per_dis[d] = K_emit
        end
        return
    end
    a = (N - 1) / 2
    base = Beta(a, a)
    y_lo = (1 + phi_keep) / 2
    td = truncated(base, y_lo, 1.0)
    p_above = 1.0 - cdf(base, y_lo)
    @inbounds for d in 1:n_dis
        target = randn(F, N); target .*= sqrtN / norm(target)
        p_cpu[:, 1, d] .= target
        K_kept = clamp(rand(Poisson(M_full * p_above)), 1, K_alloc - 1)
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

# ──────────────── Memory and chunking ────────────────
function mem_per_disorder_bytes(N::Int, K::Int)
    elems = Float64(N)*K + 3*Float64(N)*n_T + Float64(K)*n_T + 9*n_T + n_T
    return elems * sizeof(F)
end

function pick_chunk_size(N::Int, K::Int)
    per = mem_per_disorder_bytes(N, K)
    usable = MEM_BUDGET_GB * MEM_SAFETY * 1e9
    by_mem = floor(Int, usable / per)
    return clamp(by_mem, 1, N_DIS_TARGET)
end

# ──────────────── Energy + MC step ────────────────
function compute_energy_semismart!(E::CuVector{F}, x::CuArray{F,3},
                                    patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                    log_C_bulk_total::F)
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
    compute_energy_semismart!(Ep, xp, pat, ov, log_C_bulk_total)
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
const csv_out     = @sprintf("basin_stab_LSE_semismart_AAAI_Nramp_M%.1e_phikeep%.2f.csv",
                             M_TARGET, PHI_KEEP)

# Per-α: how many disorder samples are already in the CSV?
function read_disorder_progress(csv_file::String)
    !isfile(csv_file) && return Dict{String,Int}()
    counts = Dict{String,Int}()
    first_data = true
    open(csv_file, "r") do f
        for line in eachline(f)
            isempty(line) && continue
            startswith(line, "#") && continue
            if first_data; first_data = false; continue; end
            parts = split(line, ",")
            αkey  = String(parts[1])
            dis   = parse(Int, parts[5])
            counts[αkey] = max(get(counts, αkey, 0), dis)
        end
    end
    return counts
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
                parse(Int,     parts[5]))
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

# ──────────────── Per-α driver (handles disorder chunks) ────────────────
function run_alpha!(α::Float64, N::Int, phi_keep::Float64, K_alloc::Int,
                    M_full::Int, dis_start::Int, dis_target::Int, alpha_idx::Int)
    Nf = F(N)
    log_C_bulk_total = phi_keep < 0 ? F(-1e30) :
                       F(α*N + log_C_bulk_per(N, phi_keep, Float64(betanet)))

    n_dis_chunk = pick_chunk_size(N, K_alloc)
    dis_done = dis_start
    chunk_idx = 0
    while dis_done < dis_target
        chunk_idx += 1
        n_dis = min(n_dis_chunk, dis_target - dis_done)
        println("─"^76)
        @printf("α=%.3f  N=%d  M=%d  φ_keep=%+.3f  K_alloc=%d  chunk %d  n_dis=%d  (disorders %d..%d / %d)\n",
                α, N, M_full, phi_keep, K_alloc, chunk_idx, n_dis,
                dis_done+1, dis_done+n_dis, dis_target)

        Random.seed!(42 + 1000*alpha_idx + 7919*dis_done)
        print("  Sampling patterns ... "); t0 = time()
        p_cpu = zeros(F, N, K_alloc, n_dis)
        K_per_dis = zeros(Int, n_dis)
        sample_patterns!(p_cpu, K_per_dis, N, n_dis, phi_keep, K_alloc, M_full)
        K_used = maximum(K_per_dis)
        p_cpu = p_cpu[:, 1:K_used, :]
        @printf("K_max=%d  K_min=%d  avg=%d   %.1f s\n",
                maximum(K_per_dis), minimum(K_per_dis),
                round(Int, mean(K_per_dis)), time()-t0)

        n_chains = n_T * n_dis
        print("  GPU alloc ... "); t0 = time()
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
        @printf("%.1f s\n", time()-t0)

        initialise_at_target!(xa_g, tgts_g)
        initialise_at_target!(xb_g, tgts_g)
        compute_energy_semismart!(Ea_g, xa_g, pats_g, ov_g, log_C_bulk_total)
        compute_energy_semismart!(Eb_g, xb_g, pats_g, ov_g, log_C_bulk_total)
        CUDA.synchronize()

        println("  Equilibration ($N_EQ steps)…"); t0 = time()
        prog = Progress(N_EQ, desc="    eq: ")
        for _ in 1:N_EQ
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize(); @printf("    %.1f s\n", time()-t0)

        phia_g .= zero(F); phib_g .= zero(F); qs_g .= zero(F); phimax_g .= zero(F)
        println("  Sampling ($N_SAMP steps)…"); t0 = time()
        prog = Progress(N_SAMP, desc="    samp: ")
        for _ in 1:N_SAMP
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis, log_C_bulk_total)
            phia_g .+= vec(sum(tgts_g .* xa_g, dims=1)) ./ Nf
            phib_g .+= vec(sum(tgts_g .* xb_g, dims=1)) ./ Nf
            qs_g   .+= vec(sum(xa_g .* xb_g, dims=1)) ./ Nf
            compute_phi_max_retained!(phimax_g, xa_g, pats_g, ov_g, Nf)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize(); @printf("    %.1f s\n", time()-t0)

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
                        α, T_vec[j], N, K_per_dis[d], dis_done + d,
                        phia_mat[j, d], phib_mat[j, d],
                        q_mat[j, d], phimax_mat[j, d])
            end
        end

        for a in (pats_g, tgts_g, xa_g, xb_g, xp_g, ov_g,
                  Ea_g, Eb_g, Ep_g, phia_g, phib_g, qs_g, phimax_g,
                  β_g, ra_g, ss_g)
            CUDA.unsafe_free!(a)
        end
        pats_g = nothing; tgts_g = nothing
        xa_g = nothing; xb_g = nothing; xp_g = nothing
        ov_g = nothing; Ea_g = nothing; Eb_g = nothing; Ep_g = nothing
        phia_g = nothing; phib_g = nothing; qs_g = nothing; phimax_g = nothing
        β_g = nothing; ra_g = nothing; ss_g = nothing
        GC.gc(); CUDA.reclaim()

        dis_done += n_dis
    end
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("="^76)
    println("SEMISMART LSE Basin Stability — AAAI 2026  ·  N(α)-RAMP, PHI_KEEP fixed")
    @printf("  M_TARGET = %.1e   PHI_KEEP = %.3f   N(α) = round(log(M)/α)\n",
            M_TARGET, PHI_KEEP)
    @printf("  N_FLOOR = %d   N_DIS_TARGET = %d   MEM_BUDGET = %.1f GB × %.0f%% safety\n",
            N_FLOOR, N_DIS_TARGET, MEM_BUDGET_GB, 100*MEM_SAFETY)
    if n_alpha > 1
        @printf("  α grid: %.2f : %.2f : %.2f  (%d values)\n",
                alpha_vec[1], alpha_vec[2]-alpha_vec[1], alpha_vec[end], n_alpha)
    else
        @printf("  α grid: %.2f  (single value)\n", alpha_vec[1])
    end
    @printf("  T grid: %.4f : %.4f  (%d points)\n", T_vec[1], T_vec[end], n_T)
    @printf("  MC: %d eq + %d samp\n", N_EQ, N_SAMP)
    println("="^76)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Plan per α
    plan = Vector{NamedTuple}(undef, n_alpha)
    println("Per-α plan:")
    @printf("  %-6s %-4s %-12s %-12s %-9s %-12s %-12s %-7s %-5s\n",
            "α", "N", "M", "K_expected", "K/M", "K_alloc", "status", "φ_keep", "chunk")
    for i in 1:n_alpha
        α = Float64(alpha_vec[i])
        N = N_for_alpha(α)
        phi_keep, K_alloc, status, M_full, p_above = derive_K(N, α)
        K_expected = Float64(M_full) * p_above
        chunk = status == :infeasible ? 0 : pick_chunk_size(N, K_alloc)
        plan[i] = (alpha_idx=i, α=α, N=N, phi_keep=phi_keep, K_alloc=K_alloc,
                   status=status, M_full=M_full, p_above=p_above,
                   K_expected=K_expected, chunk=chunk)
        @printf("  %-6.3f %-4d %-12d %-12.4g %-9.3e %-12d %-12s %+6.3f %-5d\n",
                α, N, M_full, K_expected, p_above, K_alloc, String(status), phi_keep, chunk)
    end
    println()

    # Plan side-file (always written/refreshed; doesn't affect resume logic)
    plan_file = replace(csv_out, ".csv" => "_plan.csv")
    open(plan_file, "w") do f
        @printf(f, "# generator=basin_stab_LSE_semismart_AAAI_Nramp_phikeep.jl  M_TARGET=%.1e  PHI_KEEP=%.3f  generated=%s\n",
                M_TARGET, PHI_KEEP, string(now()))
        write(f, "alpha,N,M,K_expected,K_over_M,K_alloc,status,phi_keep\n")
        for p in plan
            @printf(f, "%.3f,%d,%d,%.6e,%.6e,%d,%s,%.4f\n",
                    p.α, p.N, p.M_full, p.K_expected, p.p_above,
                    p.K_alloc, String(p.status), p.phi_keep)
        end
    end
    println("Plan written to: ", plan_file, "\n")

    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# generator=basin_stab_LSE_semismart_AAAI_Nramp_phikeep.jl  M_TARGET=%.1e  PHI_KEEP=%.3f  N_FLOOR=%d  betanet=%s\n",
                    M_TARGET, PHI_KEEP, N_FLOOR, betanet)
            @printf(f, "# N_EQ=%d  N_SAMP=%d  N_DIS_TARGET=%d  MEM_BUDGET_GB=%.1f  MEM_SAFETY=%.2f  generated=%s\n",
                    N_EQ, N_SAMP, N_DIS_TARGET, MEM_BUDGET_GB, MEM_SAFETY, string(now()))
            # Plan rows as comments inside the main CSV (one per α)
            @printf(f, "# plan_columns: alpha,N,M,K_expected,K_over_M,K_alloc,status\n")
            for p in plan
                @printf(f, "# plan: %.3f,%d,%d,%.6e,%.6e,%d,%s\n",
                        p.α, p.N, p.M_full, p.K_expected, p.p_above,
                        p.K_alloc, String(p.status))
            end
            write(f, "alpha,T,N_used,K_retained,disorder,phi_a,phi_b,q12,phi_max_retained\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No CSV — starting fresh.")
    end
    progress = FRESH_START ? Dict{String,Int}() : read_disorder_progress(csv_out)

    for p in plan
        p.status == :infeasible && (println("Skip α=$(p.α): infeasible (K > K_HARD_CAP=$(K_HARD_CAP))"); continue)
        αkey = @sprintf("%.3f", p.α)
        dis_done = get(progress, αkey, 0)
        if dis_done >= N_DIS_TARGET
            @printf("α=%.3f already has %d ≥ %d disorders, skipping.\n", p.α, dis_done, N_DIS_TARGET)
            continue
        end
        run_alpha!(p.α, p.N, p.phi_keep, p.K_alloc, p.M_full,
                   dis_done, N_DIS_TARGET, p.alpha_idx)
        print("  Sorting CSV… "); sort_csv!(csv_out); println("done.")
    end

    println("\n" * "="^76)
    @printf("CSV: %s\n", csv_out)
    println("="^76)
end

main()

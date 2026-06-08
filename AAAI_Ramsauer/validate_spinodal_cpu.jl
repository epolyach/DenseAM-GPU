#=
CPU multi-thread semismart LSE MC for spinodal validation.

For each (α, N): paired chains initialised at ξ¹ (cold) and at a
random sphere point (hot). The cold-chain T_boundary at the largest
accessible N tracks the spinodal T_sp(α); the hot-chain T_boundary
tracks the thermodynamic equilibration boundary T_c^sd(α).

Output: spinodal_probe_cpu.csv, one row per (α, N, T, disorder, init)
with mean φ_target over the sampling window.

Usage
  julia -t auto validate_spinodal_cpu.jl
=#

using Random, Statistics, LinearAlgebra, Printf
using SpecialFunctions, QuadGK, Distributions
using Base.Threads
using Dates: now

const F = Float32
const BETANET = F(1.0)

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
    val, _ = quadgk(integrand, -1.0 + 1e-12, phi_keep; rtol=1e-8)
    log_norm = -logbeta((N - 1)/2, 0.5)
    return f_ref + log(val) + log_norm
end

# ──────────────── Pattern sampling ────────────────
function pick_phi_keep(N::Int, α::Float64, K_target::Int;
                       phi_min::Float64=0.10, phi_max::Float64=0.97)
    M_full = exp(α * N)            # Float64 — may exceed Int64 at large N
    if M_full <= Float64(K_target)
        return -1.0, M_full
    end
    lo, hi = phi_min, phi_max
    Klo = M_full * prob_phi_above(N, lo)
    Khi = M_full * prob_phi_above(N, hi)
    Khi > K_target && return phi_max, M_full   # infeasible: cap
    if Klo <= K_target
        return lo, M_full
    end
    for _ in 1:60
        mid = 0.5*(lo + hi)
        Kmid = M_full * prob_phi_above(N, mid)
        if Kmid > K_target; lo = mid; else; hi = mid; end
        (hi - lo) < 1e-4 && break
    end
    return 0.5*(lo + hi), M_full
end

function generate_patterns(N::Int, α::Float64, K_target::Int,
                           phi_keep::Float64, M_full::Float64, seed::Int)
    rng = MersenneTwister(seed)
    Nf = F(N); sqrtN = sqrt(Nf)
    target = randn(rng, F, N); target .*= sqrtN / norm(target)

    if phi_keep < 0
        # honest: M_full ≤ K_target, store M_full random patterns
        K_used = min(round(Int, M_full), K_target)
        patterns = zeros(F, N, K_used)
        patterns[:, 1] .= target
        for k in 1:K_used-1
            cand = randn(rng, F, N); cand .*= sqrtN / norm(cand)
            patterns[:, k+1] .= cand
        end
        return patterns, target, K_used
    end

    a = (N - 1) / 2
    base = Beta(a, a)
    y_lo = (1 + phi_keep) / 2
    td = truncated(base, y_lo, 1.0)
    # Use prob_phi_above (beta_inc tail) for the conditional probability —
    # cdf(base, y_lo) saturates to 1 at extreme tails, breaking 1-cdf.
    p_above = prob_phi_above(N, phi_keep)
    mean_K = M_full * p_above
    K_kept = if mean_K > 1e9
        # Poisson with huge mean: use Gaussian approximation, clamped
        clamp(round(Int, mean_K + sqrt(mean_K) * randn(rng)), 1, K_target - 1)
    else
        clamp(rand(rng, Poisson(mean_K)), 1, K_target - 1)
    end
    patterns = zeros(F, N, K_kept + 1)
    patterns[:, 1] .= target
    for k in 1:K_kept
        φ = F(2*rand(rng, td) - 1)
        perp = randn(rng, F, N)
        proj = sum(perp .* target) / Nf
        @. perp = perp - proj * target
        perp_norm = norm(perp)
        scale = F(sqrt(Float64(Nf) * (1.0 - Float64(φ)^2))) / max(perp_norm, F(1e-12))
        @. perp = perp * scale
        @views @. patterns[:, k+1] = φ * target + perp
    end
    return patterns, target, K_kept + 1
end

# ──────────────── Energy ────────────────
# H = -(1/β_net) log(Σ_k exp(β_net · pat_k · x) + exp(log_C_bulk_total))
@inline function compute_energy!(overlap::Vector{F}, x::Vector{F},
                                  patterns::Matrix{F},
                                  log_C_bulk_total::F)::F
    N = length(x)
    K = size(patterns, 2)
    @inbounds for k in 1:K
        s = zero(F)
        @simd for n in 1:N
            s += patterns[n, k] * x[n]
        end
        overlap[k] = BETANET * s
    end
    m = -F(Inf)
    @inbounds for k in 1:K
        if overlap[k] > m; m = overlap[k]; end
    end
    m_eff = m > log_C_bulk_total ? m : log_C_bulk_total
    s_retained = zero(F)
    @inbounds @simd for k in 1:K
        s_retained += exp(overlap[k] - m_eff)
    end
    s_bulk = exp(log_C_bulk_total - m_eff)
    return -(m_eff + log(s_retained + s_bulk)) / BETANET
end

# Single MC step (Metropolis on sphere). Returns updated energy.
@inline function mc_step!(x::Vector{F}, xp::Vector{F},
                          overlap::Vector{F},
                          patterns::Matrix{F},
                          E::F, Nf::F, ss::F, βchain::F,
                          log_C_bulk_total::F,
                          rng::AbstractRNG)::F
    N = length(x)
    @inbounds for n in 1:N
        xp[n] = x[n] + ss * randn(rng, F)
    end
    nrm = norm(xp)
    rescale = sqrt(Nf) / nrm
    @inbounds @simd for n in 1:N
        xp[n] *= rescale
    end
    Ep = compute_energy!(overlap, xp, patterns, log_C_bulk_total)
    if rand(rng, F) < exp(-βchain * (Ep - E))
        @inbounds @simd for n in 1:N
            x[n] = xp[n]
        end
        return Ep
    end
    return E
end

function init_hot!(x::Vector{F}, sqrtN::F, rng::AbstractRNG)
    N = length(x)
    @inbounds for n in 1:N
        x[n] = randn(rng, F)
    end
    nrm = norm(x)
    rescale = sqrtN / nrm
    @inbounds @simd for n in 1:N
        x[n] *= rescale
    end
end

# Run one chain, return mean φ_target during sampling window.
function run_chain(N::Int, T::Float64, init::Symbol,
                   patterns::Matrix{F}, target::Vector{F},
                   log_C_bulk_total::F, N_eq::Int, N_samp::Int,
                   seed::Int)::Float64
    rng = MersenneTwister(seed)
    Nf = F(N); sqrtN = sqrt(Nf)
    ss = F(2.4 * T / sqrt(N))
    βchain = F(1/T)
    K = size(patterns, 2)
    x = zeros(F, N); xp = zeros(F, N); overlap = zeros(F, K)
    if init === :cold
        @inbounds for n in 1:N; x[n] = target[n]; end
    else
        init_hot!(x, sqrtN, rng)
    end
    E = compute_energy!(overlap, x, patterns, log_C_bulk_total)
    for _ in 1:N_eq
        E = mc_step!(x, xp, overlap, patterns, E, Nf, ss, βchain,
                     log_C_bulk_total, rng)
    end
    acc = 0.0
    for _ in 1:N_samp
        E = mc_step!(x, xp, overlap, patterns, E, Nf, ss, βchain,
                     log_C_bulk_total, rng)
        s = zero(F)
        @inbounds @simd for n in 1:N
            s += target[n] * x[n]
        end
        acc += Float64(s) / N
    end
    return acc / N_samp
end

# ──────────────── Experiment ────────────────
const ALPHAS    = [0.30, 0.40, 0.50, 0.55]
const N_LIST    = [200, 500, 1000]
const T_GRID    = collect(0.05:0.025:0.55)   # 21 T values
const N_DIS     = 4
const K_TARGET  = 10_000
const N_EQ      = 8_000
const N_SAMP    = 2_000
const CSV_OUT   = "spinodal_probe_cpu.csv"

function main()
    println("="^76)
    @printf("Spinodal CPU MC: %d threads\n", nthreads())
    @printf("α = %s\n", ALPHAS)
    @printf("N = %s\n", N_LIST)
    @printf("T grid: %d points from %.3f to %.3f\n",
            length(T_GRID), first(T_GRID), last(T_GRID))
    @printf("Disorders per (α,N,T,init): %d\n", N_DIS)
    @printf("K_TARGET = %d   N_EQ = %d   N_SAMP = %d\n",
            K_TARGET, N_EQ, N_SAMP)
    println("="^76)

    # Per (α, N): φ_keep, M, bulk
    plan = Dict{Tuple{Float64,Int}, NamedTuple}()
    for α in ALPHAS, N in N_LIST
        phi_keep, M_full = pick_phi_keep(N, α, K_TARGET)
        log_C_bulk_total = phi_keep < 0 ? F(-1e30) :
                           F(α*N + log_C_bulk_per(N, phi_keep, Float64(BETANET)))
        plan[(α, N)] = (phi_keep=phi_keep, M=M_full,
                        log_C=log_C_bulk_total)
        @printf("  α=%.2f  N=%d  M=%.2e  φ_keep=%.3f  log_C=%.3f\n",
                α, N, Float64(M_full), phi_keep, Float64(log_C_bulk_total))
    end

    # Build run list
    Init = Symbol
    runs = Tuple{Float64,Int,Float64,Int,Symbol}[]
    for α in ALPHAS, N in N_LIST, T in T_GRID, d in 1:N_DIS, init in (:cold, :hot)
        push!(runs, (α, N, T, d, init))
    end
    @printf("\nTotal chains: %d\n", length(runs))

    # Patterns: shared per (α,N,d) — generate on first need per thread.
    # Simplest: thread-local cache.
    results = Vector{NamedTuple}(undef, length(runs))
    progress = Atomic{Int}(0)
    total = length(runs)
    t0 = time()

    # Pre-generate patterns per (α, N, d). They are small for our scale.
    pat_cache = Dict{Tuple{Float64,Int,Int}, Tuple{Matrix{F}, Vector{F}, Int}}()
    cache_lock = ReentrantLock()
    function get_patterns(α, N, d)
        key = (α, N, d)
        # double-checked locking
        haskey(pat_cache, key) && return pat_cache[key]
        lock(cache_lock) do
            haskey(pat_cache, key) && return
            ent = plan[(α, N)]
            seed = 12345 + 100_000*round(Int, α*100) + 1000*N + d
            patterns, target, _ = generate_patterns(N, α, K_TARGET,
                                                   ent.phi_keep, ent.M, seed)
            pat_cache[key] = (patterns, target, size(patterns, 2))
        end
        return pat_cache[key]
    end
    # Warm cache (single-threaded — fast enough)
    print("Generating patterns ... "); tp = time()
    for α in ALPHAS, N in N_LIST, d in 1:N_DIS
        get_patterns(α, N, d)
    end
    @printf("%.1f s\n", time()-tp)

    @threads for i in 1:length(runs)
        α, N, T, d, init = runs[i]
        ent = plan[(α, N)]
        patterns, target, _ = pat_cache[(α, N, d)]
        seed = 67890 + 100_000*round(Int, α*100) + 1000*N + 10*d +
               (init === :cold ? 0 : 7919) +
               round(Int, 100*T)
        phi = run_chain(N, T, init, patterns, target, ent.log_C,
                        N_EQ, N_SAMP, seed)
        results[i] = (α=α, N=N, T=T, d=d, init=String(init), phi=phi)
        n_done = atomic_add!(progress, 1) + 1
        if mod(n_done, max(1, total ÷ 40)) == 0
            elapsed = time() - t0
            eta = elapsed * (total - n_done) / max(1, n_done)
            @printf("  %d / %d  (%.1f s, ETA %.0f s)\n",
                    n_done, total, elapsed, eta)
        end
    end
    @printf("Done: %.1f s\n", time()-t0)

    # Write CSV
    open(CSV_OUT, "w") do f
        @printf(f, "# generator=validate_spinodal_cpu.jl  threads=%d  K_TARGET=%d  N_EQ=%d  N_SAMP=%d  generated=%s\n",
                nthreads(), K_TARGET, N_EQ, N_SAMP, string(now()))
        write(f, "alpha,N,T,disorder,init,phi\n")
        for r in results
            @printf(f, "%.3f,%d,%.4f,%d,%s,%.6f\n",
                    r.α, r.N, r.T, r.d, r.init, r.phi)
        end
    end
    @printf("Saved: %s\n", CSV_OUT)
end

main()

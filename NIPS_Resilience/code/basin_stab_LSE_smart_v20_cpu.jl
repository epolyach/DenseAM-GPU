#=
v20 CPU prototype — Smart-MC LSE with sampled passive-sea log C per disorder.
────────────────────────────────────────────────────────────────────────────
Goal: verify that drawing log C ~ N(μ_logC, σ²_logC) per disorder (instead of
fixing C at its mean as in v18) closes the boundary bias against honest N=25.

Multi-threaded over disorder realisations (Threads.@threads).  N=25 is small
enough that BLAS contention is negligible; we set BLAS to 1 thread so the
outer @threads owns all cores.

Algorithm per α:
  M = exp(αN)
  if M ≤ BUDGET_K  →  HONEST mode: all M patterns uniformly on √N-sphere,
                     no C term (logC = -∞).
  else             →  SMART mode:  ξ¹ + (K-1) competitors with overlap > φ_cut
                     with ξ¹, plus a log-C drawn per disorder from N(μ,σ²),
                     where μ,σ are computed from the truncated Beta integrals
                       I₁ = ∫₀^x_cut f(x) exp(-2βN(1-x)) dx     (E[Ĉ_per])
                       I₂ = ∫₀^x_cut f(x) exp(-4βN(1-x)) dx     (E[Ĉ²_per])
                       Ē  = (M-K)·I₁    Var = (M-K)·I₂
                       σ² = log(1 + Var/Ē²)   μ = log(Ē) - σ²/2

Output CSV: basin_stab_LSE_smart_v20_cpu_N<N>.csv
  alpha, T, disorder, phi, mode, logC
Header line:  # N=…  K_max=…  betanet=…  generated=…

Usage:
  julia -t auto basin_stab_LSE_smart_v20_cpu.jl           # resume
  julia -t auto basin_stab_LSE_smart_v20_cpu.jl --fresh   # overwrite
=#

using Random, Statistics, LinearAlgebra, Printf
using SpecialFunctions, QuadGK
using Base.Threads
using Dates: now

BLAS.set_num_threads(1)   # let @threads own cores

const F             = Float32
const N             = 50
const BUDGET_K      = 10_000
const betanet       = F(1.0)
const N_EQ          = 2^14   # 16384 — halved from honest_N25 for CPU prototype
const N_SAMP        = 2^12   # 4096

const MAX_N_TRIALS  = 32
const MIN_N_TRIALS  = 16
const alpha_vec     = collect(F(0.30):F(0.05):F(0.60))
const n_alpha       = length(alpha_vec)
const T_vec         = collect(F(0.025):F(0.025):F(0.500))
const n_T           = length(T_vec)
const beta_a        = (N - 1) / 2

const FRESH_START   = "--fresh" in ARGS
const CSV_OUT       = @sprintf("basin_stab_LSE_smart_v20_cpu_N%d.csv", N)

function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / max(1, n_alpha - 1)
    return max(MIN_N_TRIALS, round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t))
end
const n_disorder_vec = [n_trials_for_alpha(i) ÷ 2 for i in 1:n_alpha]

# ──────────────── Beta upper-tail tools (from v19) ────────────────
function log_beta_sf(x::Float64, a::Float64)
    x <= 0 && return 0.0
    x >= 1 && return -Inf
    if x < 0.95
        _, q = beta_inc(a, a, x); q > 0 && return log(q)
    end
    u = 1 - x
    integrand(t) = t > 0 ? exp((a - 1) * (log(t) + log1p(-u * t))) : 0.0
    val, _ = quadgk(integrand, 0.0, 1.0; rtol=1e-10)
    return a * log(u) + log(val) - logbeta(a, a)
end

function beta_isf_log(log_p::Float64, a::Float64)
    log_p >= 0 && return 0.0
    isinf(log_p) && log_p < 0 && return 1.0
    log_u = (log_p + log(a) + logbeta(a, a)) / a
    log_u = min(log_u, log(0.5))
    for _ in 1:60
        u = clamp(exp(log_u), 1e-300, 0.5); x = 1 - u
        lp = log_beta_sf(x, a); err = lp - log_p
        abs(err) < 1e-10 && break
        log_pdf = (a - 1) * (log(x) + log(u)) - logbeta(a, a)
        dlog = exp(log_pdf + log(u) - lp); dlog < 1e-12 && (dlog = a)
        log_u -= err / dlog
    end
    return 1 - exp(log_u)
end

function phi_cut_at(α::Float64, K::Int)
    log_p_keep = log(K) - α * N
    log_p_keep >= 0 && return -1.0
    return 2 * beta_isf_log(log_p_keep, Float64(beta_a)) - 1
end

# ──────────────── Passive-sea moments (truncated to φ < φ_cut) ────────────────
function C_moments(α::Float64, K::Int, βn::Float64, φ_cut::Float64)
    M = exp(α * N)
    a = Float64(beta_a)
    x_cut = (1 + φ_cut) / 2
    function I_k(k::Int)
        f(x) = begin
            (x <= 0 || x >= 1) && return 0.0
            log_pdf = (a - 1) * (log(x) + log1p(-x)) - logbeta(a, a)
            exp(-2 * k * βn * N * (1 - x) + log_pdf)
        end
        val, _ = quadgk(f, 0.0, x_cut; rtol=1e-10)
        return val
    end
    mean_C = (M - K) * I_k(1)
    var_C  = (M - K) * I_k(2)
    return mean_C, var_C
end

# ──────────────── Quantile table for smart-mode pattern draws ────────────────
struct QTable
    lo::Float64; hi::Float64; n::Int; x::Vector{Float64}
end

function build_qtable(α::Float64; n_grid=1000, depth=200.0)
    hi = log(BUDGET_K) - α * N
    lo = hi - depth
    a = Float64(beta_a)
    xs = [beta_isf_log(la, a) for la in range(lo, hi, length=n_grid)]
    return QTable(lo, hi, n_grid, xs)
end

@inline function sample_phi(q::QTable, log_u::Float64)
    la = max(q.hi + log_u, q.lo)
    frac = (la - q.lo) / (q.hi - q.lo)
    idx_f = frac * (q.n - 1) + 1
    i = clamp(floor(Int, idx_f), 1, q.n - 1); w = idx_f - i
    return 2 * (q.x[i] * (1 - w) + q.x[i + 1] * w) - 1
end

# ──────────────── Pattern generation ────────────────
function gen_honest_patterns(M::Int, rng::AbstractRNG)
    P = randn(rng, F, N, M)
    @inbounds for μ in 1:M
        c = @view P[:, μ]
        nm = norm(c); nm > 0 && (c .*= sqrt(F(N)) / nm)
    end
    return P
end

function gen_smart_patterns(K::Int, qt::QTable, rng::AbstractRNG)
    P = Matrix{F}(undef, N, K)
    ξ1 = randn(rng, Float64, N); ξ1 .*= sqrt(N) / norm(ξ1)
    @inbounds for i in 1:N; P[i, 1] = F(ξ1[i]); end
    u_ref = ξ1 ./ sqrt(N)
    g = zeros(N)
    @inbounds for μ in 2:K
        log_u = log(rand(rng))
        φ_μ = sample_phi(qt, log_u)
        randn!(rng, g)
        g .-= dot(g, u_ref) .* u_ref
        nm = norm(g)
        if nm < 1e-12
            randn!(rng, g); g .-= dot(g, u_ref) .* u_ref; nm = norm(g)
        end
        g ./= nm
        s = sqrt(N * (1 - φ_μ^2))
        for i in 1:N
            P[i, μ] = F(φ_μ * ξ1[i] + s * g[i])
        end
    end
    return P
end

# ──────────────── LSE energy with constant logC ────────────────
@inline function energy_lse(x::Vector{F}, P::Matrix{F}, logC::F, βn::F,
                            args::Vector{F})
    Nf = F(N)
    K = size(P, 2)
    @inbounds for μ in 1:K
        ov = zero(F)
        for i in 1:N
            ov += P[i, μ] * x[i]
        end
        args[μ] = -βn * (Nf - ov)
    end
    args[K + 1] = logC
    m = -F(Inf)
    @inbounds for k in 1:(K + 1)
        a = args[k]; a > m && (m = a)
    end
    s = zero(F)
    @inbounds for k in 1:(K + 1)
        s += exp(args[k] - m)
    end
    return -(m + log(s)) / βn
end

# ──────────────── MC chain (returns mean φ over sampling phase) ────────────────
function run_chain(P::Matrix{F}, logC::F, T::F, n_eq::Int, n_samp::Int, rng::AbstractRNG)
    Nf = F(N)
    βn = betanet
    β  = F(1) / T
    ss = F(2.4) * T / sqrt(Nf)
    K  = size(P, 2)
    args = Vector{F}(undef, K + 1)

    x  = Vector{F}(undef, N)
    @inbounds for i in 1:N; x[i] = P[i, 1]; end
    E  = energy_lse(x, P, logC, βn, args)
    xp = similar(x)

    for _ in 1:n_eq
        @inbounds for i in 1:N
            xp[i] = x[i] + ss * F(randn(rng))
        end
        nm = norm(xp); nm > 0 && (xp .*= sqrt(Nf) / nm)
        Ep = energy_lse(xp, P, logC, βn, args)
        if rand(rng, F) < exp(-β * (Ep - E))
            copyto!(x, xp); E = Ep
        end
    end

    phi_acc = zero(F)
    for _ in 1:n_samp
        @inbounds for i in 1:N
            xp[i] = x[i] + ss * F(randn(rng))
        end
        nm = norm(xp); nm > 0 && (xp .*= sqrt(Nf) / nm)
        Ep = energy_lse(xp, P, logC, βn, args)
        if rand(rng, F) < exp(-β * (Ep - E))
            copyto!(x, xp); E = Ep
        end
        φ = zero(F)
        @inbounds for i in 1:N
            φ += P[i, 1] * x[i]
        end
        phi_acc += φ / Nf
    end
    return phi_acc / F(n_samp)
end

# ──────────────── Resume helpers ────────────────
function read_completed(csv_file::String)
    !isfile(csv_file) && return Set{String}()
    s = Set{String}()
    for l in eachline(csv_file)
        (isempty(l) || startswith(l, "#") || startswith(l, "alpha")) && continue
        push!(s, String(split(l, ",")[1]))
    end
    return s
end

# ──────────────── Main ────────────────
function main()
    println("=" ^ 72)
    println("Smart-MC LSE v20 CPU prototype  (sampled passive-sea log C)")
    @printf("N=%d   K_max=%d   β_net=%.2f\n", N, BUDGET_K, betanet)
    @printf("α grid: %s\n", join([@sprintf("%.2f", α) for α in alpha_vec], ", "))
    @printf("T grid: %d points  [%.3f, %.3f]\n", n_T, first(T_vec), last(T_vec))
    @printf("MC: N_EQ=%d   N_SAMP=%d\n", N_EQ, N_SAMP)
    @printf("Threads: %d\n", nthreads())
    println("=" ^ 72)

    if FRESH_START || !isfile(CSV_OUT)
        open(CSV_OUT, "w") do f
            @printf(f, "# N=%d  K_max=%d  betanet=%.4f  generated=%s\n",
                    N, BUDGET_K, betanet, string(now()))
            write(f, "alpha,T,disorder,phi,mode,logC\n")
        end
    end
    completed = FRESH_START ? Set{String}() : read_completed(CSV_OUT)

    t_total = 0.0
    for (gi, α) in enumerate(alpha_vec)
        key = @sprintf("%.3f", α)
        if key in completed
            @printf("Skip α=%s  (already in CSV)\n", key); continue
        end
        α64 = Float64(α)
        M_real = exp(α64 * N)
        M_int  = min(round(Int, M_real), 10_000_000)
        n_dis  = n_disorder_vec[gi]

        if M_int <= BUDGET_K
            K_eff = M_int
            mode  = "honest"
            μ_logC = -F(30); σ_logC = zero(F)
            qt = nothing
            @printf("\nα=%.3f  M=%d ≤ K=%d  →  HONEST mode\n", α, M_int, BUDGET_K)
        else
            K_eff = BUDGET_K
            φ_cut = phi_cut_at(α64, K_eff)
            mean_C, var_C = C_moments(α64, K_eff, Float64(betanet), φ_cut)
            σ²_logC = log(1 + var_C / mean_C^2)
            μ_logC  = F(log(mean_C) - σ²_logC / 2)
            σ_logC  = F(sqrt(max(0.0, σ²_logC)))
            mode    = "smart"
            qt = build_qtable(α64)
            @printf("\nα=%.3f  M=%.3e  φ_cut=%.4f  Ē=%.3e  σ_logC=%.3f  →  SMART mode\n",
                    α, M_real, φ_cut, mean_C, σ_logC)
        end
        @printf("  n_dis=%d   K_eff=%d   threads=%d\n", n_dis, K_eff, nthreads())

        results = Vector{Vector{F}}(undef, n_dis)
        logCs   = Vector{F}(undef, n_dis)

        t0 = time()
        @threads for d in 1:n_dis
            rng = Xoshiro(UInt64(42 + 1000 * gi + d))
            P = mode == "honest" ? gen_honest_patterns(K_eff, rng) :
                                   gen_smart_patterns(K_eff, qt, rng)
            logC_d = mode == "smart" ? (μ_logC + σ_logC * F(randn(rng))) : F(-30)
            logCs[d] = logC_d
            res = Vector{F}(undef, n_T)
            for j in 1:n_T
                rng_chain = Xoshiro(hash((UInt64(42), UInt64(gi), UInt64(d), UInt64(j))))
                res[j] = run_chain(P, logC_d, T_vec[j], N_EQ, N_SAMP, rng_chain)
            end
            results[d] = res
        end
        t_α = time() - t0; t_total += t_α
        @printf("  elapsed: %.1f s\n", t_α)

        open(CSV_OUT, "a") do f
            for d in 1:n_dis, j in 1:n_T
                @printf(f, "%.3f,%.5f,%d,%.6f,%s,%.4f\n",
                        α, T_vec[j], d, results[d][j], mode, logCs[d])
            end
        end
        flush(stdout)
    end

    @printf("\nTotal: %.1f s   CSV: %s\n", t_total, CSV_OUT)
end

main()

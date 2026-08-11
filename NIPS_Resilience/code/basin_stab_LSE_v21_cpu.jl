#=
v21 CPU prototype — sticky-cache LSE.

Cone follows current chain x.  Patterns are sampled on demand from the
spherical-density upper tail conditional on overlap > φ_cut with current x.
Discovered patterns enter a saved list and keep their identity for the rest
of the run (they may leave the cone as the chain wanders, but are still there
if the chain returns).

Live set at each step = saved patterns whose overlap with current x exceeds
φ_cut.  Top up to K_target if the live count falls below.  Bulk (M − n_live)
patterns contribute a deterministic constant C = (M − n_live) · μ_bulk_per
inside the LSE log-sum-exp.

Usage:
  julia -t auto basin_stab_LSE_v21_cpu.jl [--N 50] [--K 1000] [--fresh]

Output CSV: basin_stab_LSE_v21_cpu_N<N>_K<K>.csv
  cols: alpha, T, disorder, phi, n_saved_final
=#

using Random, Statistics, LinearAlgebra, Printf
using SpecialFunctions, QuadGK
using Base.Threads
using Dates: now

BLAS.set_num_threads(1)

const N = let
    idx = findfirst(==("--N"), ARGS)
    idx === nothing ? 50 : parse(Int, ARGS[idx + 1])
end
const K_TARGET = let
    idx = findfirst(==("--K"), ARGS)
    idx === nothing ? 1000 : parse(Int, ARGS[idx + 1])
end
const FRESH_START = "--fresh" in ARGS

const F             = Float32
const betanet       = F(1.0)
const N_EQ          = 2^14
const N_SAMP        = 2^12

const MAX_N_TRIALS  = 32
const MIN_N_TRIALS  = 16
const alpha_vec     = collect(F(0.30):F(0.05):F(0.60))
const n_alpha       = length(alpha_vec)
const T_vec         = collect(F(0.025):F(0.025):F(0.500))
const n_T           = length(T_vec)
const beta_a        = (N - 1) / 2

const CSV_OUT = @sprintf("basin_stab_LSE_v21_cpu_N%d_K%d.csv", N, K_TARGET)

# ─── progress / instrumentation knobs ───
const STDOUT_LOCK         = ReentrantLock()
const CHAIN_PROG_EVERY    = 2048   # in-chain log every this many MC steps
const VERBOSE_DISORDER_ID = 1      # only this disorder emits in-chain progress

function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / max(1, n_alpha - 1)
    return max(MIN_N_TRIALS, round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t))
end
const n_disorder_vec = [n_trials_for_alpha(i) ÷ 2 for i in 1:n_alpha]

# ──────────────── Beta tools ────────────────
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
    log_u = clamp(log_u, -300.0, log(0.5))
    for _ in 1:60
        u = clamp(exp(log_u), 1e-300, 0.5); x = 1 - u
        lp = log_beta_sf(x, a); err = lp - log_p
        abs(err) < 1e-10 && break
        log_pdf = (a - 1) * (log(x) + log(u)) - logbeta(a, a)
        dlog = exp(log_pdf + log(u) - lp); dlog < 1e-12 && (dlog = a)
        step = err / dlog
        # damped Newton with hard clamping — overshoot was producing log_u > 0
        # → x = 1-exp(log_u) < 0 → φ_cut < -1 and sqrt(1-φ²) failed.
        step = clamp(step, -2.0, 2.0)
        log_u = clamp(log_u - step, -300.0, log(0.5))
    end
    return 1 - exp(log_u)
end

function phi_cut_at(α::Float64, K::Int)
    log_p_keep = log(K) - α * N
    log_p_keep >= 0 && return -1.0
    return 2 * beta_isf_log(log_p_keep, Float64(beta_a)) - 1
end

struct QTable
    lo::Float64; hi::Float64; n::Int; x::Vector{Float64}
end

function build_qtable(α::Float64; n_grid=1000, depth=200.0)
    hi = log(K_TARGET) - α * N
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

# Per-pattern mean bulk weight  E[exp(-βN(1-φ)) | φ < φ_cut]
function bulk_mean_per(α::Float64, K::Int, βn::Float64, φ_cut::Float64)
    a = Float64(beta_a); x_cut = (1 + φ_cut) / 2
    f(x) = (x <= 0 || x >= 1) ? 0.0 : begin
        log_pdf = (a - 1) * (log(x) + log1p(-x)) - logbeta(a, a)
        exp(-2 * βn * N * (1 - x) + log_pdf)
    end
    val, _ = quadgk(f, 0.0, x_cut; rtol=1e-10)
    return val
end

# Generate one pattern with overlap > φ_cut wrt x_ref, perpendicular direction uniform.
@inline function gen_pattern_cone!(out::AbstractVector{F},
                                    x_ref::AbstractVector{F},
                                    qt::QTable, rng::AbstractRNG,
                                    g_buf::Vector{Float64})
    Nf = N
    φ = sample_phi(qt, log(rand(rng)))
    randn!(rng, g_buf)
    inv_sqrt_N = 1.0 / sqrt(Nf)
    # project g_buf perpendicular to x_ref/|x_ref| (|x_ref| = √N)
    dot_g_uref = 0.0
    @inbounds for i in 1:Nf
        dot_g_uref += g_buf[i] * Float64(x_ref[i]) * inv_sqrt_N
    end
    @inbounds for i in 1:Nf
        g_buf[i] -= dot_g_uref * Float64(x_ref[i]) * inv_sqrt_N
    end
    nm = 0.0
    @inbounds for i in 1:Nf; nm += g_buf[i]^2; end
    nm = sqrt(nm)
    if nm < 1e-12
        randn!(rng, g_buf)
        dot_g_uref = 0.0
        @inbounds for i in 1:Nf
            dot_g_uref += g_buf[i] * Float64(x_ref[i]) * inv_sqrt_N
        end
        @inbounds for i in 1:Nf
            g_buf[i] -= dot_g_uref * Float64(x_ref[i]) * inv_sqrt_N
        end
        nm = 0.0
        @inbounds for i in 1:Nf; nm += g_buf[i]^2; end
        nm = sqrt(nm)
    end
    inv_nm = 1.0 / nm
    s = sqrt(Nf * (1 - φ^2))
    @inbounds for i in 1:Nf
        out[i] = F(φ * Float64(x_ref[i]) + s * g_buf[i] * inv_nm)
    end
end

# LSE with live patterns + bulk C
function compute_lse(ovs::AbstractVector{F}, n_saved::Int, φ_cut::F,
                     M_real::Float64, μ_bulk_per::Float64, βn::F, Nf::F)
    n_live = 0
    m = -F(Inf)
    @inbounds for i in 1:n_saved
        if ovs[i] > φ_cut
            a = -βn * (Nf - ovs[i])
            a > m && (m = a)
            n_live += 1
        end
    end
    C_val = (M_real - Float64(n_live)) * μ_bulk_per
    logC = C_val > 0 ? F(log(C_val)) : -F(30)
    logC > m && (m = logC)
    s = zero(F)
    @inbounds for i in 1:n_saved
        if ovs[i] > φ_cut
            s += exp(-βn * (Nf - ovs[i]) - m)
        end
    end
    s += exp(logC - m)
    return -(m + log(s)) / βn
end

# Single chain
function run_chain(target::Vector{F}, K_target::Int, φ_cut::F, M_real::Float64,
                   μ_bulk_per::Float64, qt::QTable,
                   T::F, n_eq::Int, n_samp::Int, rng::AbstractRNG;
                   verbose_label::Union{Nothing,String}=nothing)
    Nf = F(N)
    βn = betanet
    β  = F(1) / T
    ss = F(2.4) * T / sqrt(Nf)
    t_chain_start = time()
    n_acc = 0; n_prop = 0
    total_steps = n_eq + n_samp

    capacity = max(2 * K_target, 2000)
    pats   = Matrix{F}(undef, N, capacity)
    ovs_x  = Vector{F}(undef, capacity)
    ovs_xp = Vector{F}(undef, capacity)
    g_buf  = Vector{Float64}(undef, N)

    # Pattern 1 = ξ¹; patterns 2..K_target drawn from cone around ξ¹.
    n_saved = 1
    @inbounds for i in 1:N; pats[i, 1] = target[i]; end
    for _ in 2:K_target
        n_saved += 1
        @views gen_pattern_cone!(pats[:, n_saved], target, qt, rng, g_buf)
    end

    x = copy(target)
    @views mul!(ovs_x[1:n_saved], pats[:, 1:n_saved]', x)
    @views ovs_x[1:n_saved] ./= Nf

    E_x = compute_lse(view(ovs_x, 1:n_saved), n_saved, φ_cut, M_real, μ_bulk_per, βn, Nf)

    xp = similar(x)
    phi_sum = zero(F)

    for step in 1:(n_eq + n_samp)
        @inbounds for i in 1:N
            xp[i] = x[i] + ss * F(randn(rng))
        end
        nm = norm(xp); nm > 0 && (xp .*= sqrt(Nf) / nm)

        @views mul!(ovs_xp[1:n_saved], pats[:, 1:n_saved]', xp)
        @views ovs_xp[1:n_saved] ./= Nf

        n_live_xp = 0
        @inbounds for i in 1:n_saved
            ovs_xp[i] > φ_cut && (n_live_xp += 1)
        end

        if n_live_xp < K_target
            deficit = K_target - n_live_xp
            for _ in 1:deficit
                n_saved += 1
                if n_saved > capacity
                    new_capacity = 2 * capacity
                    new_pats = Matrix{F}(undef, N, new_capacity)
                    @views new_pats[:, 1:capacity] .= pats
                    pats = new_pats
                    new_ovs_x = Vector{F}(undef, new_capacity)
                    @views new_ovs_x[1:capacity] .= ovs_x
                    ovs_x = new_ovs_x
                    new_ovs_xp = Vector{F}(undef, new_capacity)
                    @views new_ovs_xp[1:capacity] .= ovs_xp
                    ovs_xp = new_ovs_xp
                    capacity = new_capacity
                end
                @views gen_pattern_cone!(pats[:, n_saved], xp, qt, rng, g_buf)
                ov_xp = zero(F); ov_x = zero(F)
                @inbounds for i in 1:N
                    ov_xp += pats[i, n_saved] * xp[i]
                    ov_x  += pats[i, n_saved] * x[i]
                end
                ovs_xp[n_saved] = ov_xp / Nf
                ovs_x[n_saved]  = ov_x / Nf
            end
        end

        E_xp = compute_lse(view(ovs_xp, 1:n_saved), n_saved, φ_cut, M_real, μ_bulk_per, βn, Nf)

        n_prop += 1
        if rand(rng, F) < exp(-β * (E_xp - E_x))
            copyto!(x, xp)
            ovs_x, ovs_xp = ovs_xp, ovs_x
            E_x = E_xp
            n_acc += 1
        end

        if step > n_eq
            φ_inst = zero(F)
            @inbounds for i in 1:N
                φ_inst += target[i] * x[i]
            end
            phi_sum += φ_inst / Nf
        end

        if verbose_label !== nothing && (step % CHAIN_PROG_EVERY == 0 || step == total_steps)
            t_now = time() - t_chain_start
            steps_per_s = step / max(t_now, 1e-9)
            eta_chain  = (total_steps - step) / max(steps_per_s, 1e-9)
            φ_now = zero(F)
            @inbounds for i in 1:N
                φ_now += target[i] * x[i]
            end
            φ_now /= Nf
            lock(STDOUT_LOCK) do
                @printf("    [%s] step=%5d/%d  n_saved=%9d  acc=%.3f  φ=%+.3f  %.0f st/s  chain-elapsed=%.0fs  chain-ETA=%.0fs\n",
                        verbose_label, step, total_steps,
                        n_saved,
                        n_prop > 0 ? n_acc / n_prop : 0.0,
                        φ_now, steps_per_s, t_now, eta_chain)
                flush(stdout)
            end
        end
    end

    return phi_sum / F(n_samp), n_saved
end

function read_completed(csv_file::String)
    !isfile(csv_file) && return Set{String}()
    s = Set{String}()
    for l in eachline(csv_file)
        (isempty(l) || startswith(l, "#") || startswith(l, "alpha")) && continue
        push!(s, String(split(l, ",")[1]))
    end
    return s
end

function main()
    println("=" ^ 72)
    println("v21 CPU prototype — sticky-cache cone follows chain x")
    @printf("N=%d   K_target=%d   β_net=%.2f\n", N, K_TARGET, betanet)
    @printf("α grid: %s\n", join([@sprintf("%.2f", α) for α in alpha_vec], ", "))
    @printf("T grid: %d pts  [%.3f, %.3f]\n", n_T, first(T_vec), last(T_vec))
    @printf("MC: N_EQ=%d  N_SAMP=%d\n", N_EQ, N_SAMP)
    @printf("Threads: %d\n", nthreads())
    println("=" ^ 72)

    if FRESH_START || !isfile(CSV_OUT)
        open(CSV_OUT, "w") do f
            @printf(f, "# N=%d  K_target=%d  betanet=%.4f  generated=%s\n",
                    N, K_TARGET, betanet, string(now()))
            write(f, "alpha,T,disorder,phi,n_saved_final\n")
        end
    end
    completed = FRESH_START ? Set{String}() : read_completed(CSV_OUT)

    t_total = 0.0
    for (gi, α) in enumerate(alpha_vec)
        key = @sprintf("%.3f", α)
        if key in completed
            @printf("Skip α=%s (already in CSV)\n", key); continue
        end
        α64 = Float64(α)
        M_real = exp(α64 * N)
        n_dis = n_disorder_vec[gi]

        if M_real <= 2 * K_TARGET
            @printf("\nα=%.3f  M=%.2e ≤ 2·K=%d  — skipping (smart-MC meaningful only when M ≫ K)\n",
                    α, M_real, 2 * K_TARGET)
            continue
        end

        φ_cut_val = F(phi_cut_at(α64, K_TARGET))
        if φ_cut_val <= F(-1) || φ_cut_val >= F(1)
            @printf("\nα=%.3f  φ_cut=%.3f out of (-1, 1) — skipping degenerate cone\n",
                    α, φ_cut_val)
            continue
        end
        μ_bulk_per = bulk_mean_per(α64, K_TARGET, Float64(betanet), Float64(φ_cut_val))
        qt = build_qtable(α64)

        @printf("\nα=%.3f  M=%.2e  φ_cut=%.4f  μ_bulk_per=%.3e\n",
                α, M_real, φ_cut_val, μ_bulk_per)
        @printf("  n_dis=%d  threads=%d\n", n_dis, nthreads())

        results = Vector{Vector{F}}(undef, n_dis)
        n_saved_final = Vector{Vector{Int}}(undef, n_dis)

        total_chains = n_dis * n_T
        n_done = Threads.Atomic{Int}(0)
        t0 = time()
        @threads for d in 1:n_dis
            rng = Xoshiro(UInt64(42 + 1000 * gi + d))
            target = randn(rng, F, N)
            target .*= sqrt(F(N)) / norm(target)

            res_phi = Vector{F}(undef, n_T)
            res_nsav = Vector{Int}(undef, n_T)
            for j in 1:n_T
                rng_chain = Xoshiro(hash((UInt64(42), UInt64(gi), UInt64(d), UInt64(j))))
                label = (d == VERBOSE_DISORDER_ID) ?
                    @sprintf("α=%.3f d=%d T=%.3f", α, d, T_vec[j]) : nothing
                t_chain = time()
                res_phi[j], res_nsav[j] = run_chain(
                    target, K_TARGET, φ_cut_val, M_real, μ_bulk_per, qt,
                    T_vec[j], N_EQ, N_SAMP, rng_chain;
                    verbose_label=label)
                done = Threads.atomic_add!(n_done, 1) + 1
                t_chain_dt = time() - t_chain
                elapsed_α   = time() - t0
                rate        = done / max(elapsed_α, 1e-9)
                eta_α       = (total_chains - done) / max(rate, 1e-9)
                lock(STDOUT_LOCK) do
                    @printf("  [%3d/%3d] α=%.3f  d=%2d T=%.3f  φ=%+.4f  n_saved=%9d  chain=%5.1fs  α-elapsed=%6.0fs  α-ETA=%6.0fs\n",
                            done, total_chains, α, d, T_vec[j],
                            res_phi[j], res_nsav[j],
                            t_chain_dt, elapsed_α, eta_α)
                    flush(stdout)
                end
            end
            results[d] = res_phi
            n_saved_final[d] = res_nsav
        end
        t_α = time() - t0; t_total += t_α
        max_nsav = maximum(maximum.(n_saved_final))
        mean_nsav = mean(maximum.(n_saved_final))
        @printf("  α=%.3f  done: elapsed=%.1f s   max(n_saved)=%d   mean(n_saved)=%.0f   total-so-far=%.0fs\n",
                α, t_α, max_nsav, mean_nsav, t_total)

        open(CSV_OUT, "a") do f
            for d in 1:n_dis, j in 1:n_T
                @printf(f, "%.3f,%.5f,%d,%.6f,%d\n",
                        α, T_vec[j], d, results[d][j], n_saved_final[d][j])
            end
        end
        flush(stdout)
    end
    @printf("\nTotal: %.1f s   CSV: %s\n", t_total, CSV_OUT)
end

main()

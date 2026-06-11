#=
N-dim hybrid MC for LSE basin escape near the saddle-dominance tip.
Per (α, T, disorder, N) cell:
  Cusp φ_c = α + g_max  (β_net = 1 throughout).
  K_cusp ≈ M · Pr(φ_{1μ} > φ_c) patterns kept explicitly:
    φ_{1μ} ~ truncated Beta((N-1)/2, (N-1)/2) above (1 + φ_c)/2
    Orthogonal direction u_μ uniform on (N-1)-sphere perp to ξ¹
    Pattern  ξ^μ = √N · (φ_{1μ},  √(1 - φ_{1μ}²) · u_μ)
  Convention: ξ¹ = (√N, 0, …, 0), so φ_1(x) = x[1] / √N.

  Energy:  H(x)  = − log[ Σ_kept exp(ξ^μ · x) + M · exp(N · g_max) + exp(ξ¹ · x) ]
  Metropolis on x ∈ ℝ^N projected to |x| = √N at every step.

Output: hybrid_Ndim_LSE_N{N}.csv  in this directory.
Resume on re-run; pass --fresh to overwrite.
=#

using Base.Threads
using Distributions
using LinearAlgebra
using Random
using Printf
using Dates

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(φ_star) + φ_star

phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

# ───────── Grid & config ─────────
const N            = 100
const ALPHA_GRID   = collect(0.50:0.01:0.62)
const T_GRID       = collect(0.0025:0.005:0.0975)
const N_DIS        = 32
const N_EQ         = 10_000
const N_SAMP       = 100_000
const σ_FLOOR      = 0.01

const OUT_DIR      = @__DIR__
const STD_NORMAL   = Normal()

# ───────── K(α, N) via cusp rule, spherical Beta tail ─────────
function compute_K(α::Float64, N::Int)
    φ_c   = α + g_max
    half  = (N - 1) / 2
    log_comp = logccdf(Beta(half, half), (1.0 + φ_c) / 2.0)
    log_K = N * α + log_comp
    log_K > 40.0 && return -1
    return max(round(Int, exp(log_K)), 1)
end

# ───────── Build pattern matrix ξ^μ on the √N sphere ─────────
# Returns K × N matrix; row μ is ξ^μ.  ξ¹ = (√N, 0, …, 0) implicit.
function build_patterns(α::Float64, N::Int, K::Int, rng::AbstractRNG)
    φ_c   = α + g_max
    half  = (N - 1) / 2
    tn = truncated(Beta(half, half), (1.0 + φ_c) / 2.0, 1.0)
    sN = sqrt(N)
    P = Matrix{Float64}(undef, K, N)
    u = Vector{Float64}(undef, N - 1)
    @inbounds for k in 1:K
        φ_1μ = 2.0 * rand(rng, tn) - 1.0
        P[k, 1] = sN * φ_1μ
        # orthogonal direction uniform on (N-1)-sphere
        randn!(rng, u)
        u_norm = norm(u)
        scale  = sN * sqrt(max(0.0, 1.0 - φ_1μ * φ_1μ)) / u_norm
        for i in 2:N
            P[k, i] = scale * u[i - 1]
        end
    end
    return P
end

# ───────── log Z(x) ─────────
# log[ exp(ξ¹·x) + Σ_kept exp(ξ^μ·x) + M·exp(N·g_max) ]
@inline function log_Z(dots::Vector{Float64}, retr::Float64, log_bulk::Float64)
    max_val = max(retr, log_bulk)
    if !isempty(dots)
        @inbounds @simd for v in dots
            max_val = max(max_val, v)
        end
    end
    s = exp(retr - max_val) + exp(log_bulk - max_val)
    @inbounds @simd for v in dots
        s += exp(v - max_val)
    end
    return max_val + log(s)
end

# ───────── Metropolis on x ∈ √N · S^{N-1} ─────────
function run_mc(α::Float64, T::Float64, N::Int, P::Matrix{Float64},
                log_bulk::Float64, rng::AbstractRNG)
    K = size(P, 1)
    sN = sqrt(N)
    # Initial x at the basin: x[1] = √N · φ_eq(T), rest uniform on perp sphere
    a = min(phi_eq(T), 0.999)
    x = zeros(N)
    x[1] = sN * a
    u = randn(rng, N - 1)
    u ./= norm(u)
    scale = sN * sqrt(max(0.0, 1.0 - a * a))
    @inbounds for i in 2:N
        x[i] = scale * u[i - 1]
    end

    dots = P * x                          # K-vector
    retr = sN * x[1]                      # ξ¹ · x  (ξ¹ = √N e_1)
    logZ = log_Z(dots, retr, log_bulk)

    σ = max(σ_FLOOR, 2.0 * sqrt(T / N))
    η = Vector{Float64}(undef, N)
    x_new = Vector{Float64}(undef, N)
    dots_new = Vector{Float64}(undef, K)

    sum_φ  = 0.0
    sum_φ2 = 0.0
    n_acc  = 0
    n_steps = N_EQ + N_SAMP
    @inbounds for step in 1:n_steps
        randn!(rng, η)
        @simd for i in 1:N
            x_new[i] = x[i] + σ * η[i]
        end
        nrm = norm(x_new)
        @simd for i in 1:N
            x_new[i] *= sN / nrm
        end
        mul!(dots_new, P, x_new)
        retr_new = sN * x_new[1]
        logZ_new = log_Z(dots_new, retr_new, log_bulk)

        log_acc = (logZ_new - logZ) / T
        if log_acc >= 0.0 || log(rand(rng)) < log_acc
            copyto!(x, x_new)
            copyto!(dots, dots_new)
            retr = retr_new
            logZ = logZ_new
            n_acc += 1
        end

        if step > N_EQ
            φ_1 = x[1] / sN
            sum_φ  += φ_1
            sum_φ2 += φ_1 * φ_1
        end
    end

    mean_φ = sum_φ  / N_SAMP
    var_φ  = sum_φ2 / N_SAMP - mean_φ * mean_φ
    return mean_φ, var_φ, n_acc / n_steps
end

# ───────── Resume ─────────
function load_existing(csv_path::String)
    done = Set{Tuple{Float64,Float64,Int}}()
    isfile(csv_path) || return done
    for line in eachline(csv_path)
        (isempty(line) || startswith(line, "#") || startswith(line, "alpha")) && continue
        f = split(line, ",")
        push!(done, (parse(Float64, f[1]), parse(Float64, f[2]), parse(Int, f[3])))
    end
    return done
end

function main()
    fresh = "--fresh" in ARGS
    csv_path = joinpath(OUT_DIR, @sprintf("hybrid_Ndim_LSE_N%d.csv", N))
    if fresh && isfile(csv_path); rm(csv_path); end
    done = load_existing(csv_path)
    @info "N=$N  resume cells: $(length(done))"

    new_file = !isfile(csv_path)
    io = open(csv_path, "a")
    if new_file
        println(io,
            "# generator=hybrid_Ndim_LSE.jl  N=$N  N_DIS=$N_DIS  N_EQ=$N_EQ  N_SAMP=$N_SAMP  σ_floor=$σ_FLOOR  generated=$(Dates.now())")
        println(io, "alpha,T,disorder,K,phi_mean,phi_var,acc_rate")
        flush(io)
    end
    io_lock = ReentrantLock()

    tasks = vec([(d, α) for d in 1:N_DIS, α in ALPHA_GRID])
    counter = Threads.Atomic{Int}(0)
    total   = length(tasks)

    @threads for task in tasks
        d, α = task
        if all((α, T, d) in done for T in T_GRID)
            Threads.atomic_add!(counter, 1)
            continue
        end
        K = compute_K(α, N)
        if K < 0
            @warn "K too large, skip (α=$α)"
            Threads.atomic_add!(counter, 1)
            continue
        end
        rng_pat = MersenneTwister(hash((:Ndim_pat, N, α, d)))
        P = build_patterns(α, N, K, rng_pat)
        log_bulk = N * α + N * g_max     # M · exp(N · g_max)

        for T in T_GRID
            (α, T, d) in done && continue
            rng_mc = MersenneTwister(hash((:Ndim_mc, N, α, T, d)))
            mφ, vφ, acc = run_mc(α, T, N, P, log_bulk, rng_mc)
            lock(io_lock) do
                @printf(io, "%.3f,%.5f,%d,%d,%.6f,%.6f,%.4f\n",
                        α, T, d, K, mφ, vφ, acc)
                flush(io)
            end
        end
        n = Threads.atomic_add!(counter, 1) + 1
        if n % max(1, total ÷ 20) == 0
            @info "N=$N  $(n)/$(total)"
        end
    end
    close(io)
    @info "done → $csv_path"
end

main()

#=
First-passage escape time τ from the LSE retrieval basin under the
N-dim hybrid MC.  Mirrors escape_time_hybrid.jl but with Metropolis on
the full spin x ∈ √N · S^{N-1}, not on the 1D effective φ_1.

Per (α, T, disorder, N) cell:
  Start x at φ_1 = φ_eq(T), orthogonal sector uniform on the perp sphere.
  Metropolis on x; record step at which φ_1 = x[1]/√N first crosses 0.5.
  Cap at MAX_STEPS.

Output: escape_time_Ndim_N{N}.csv
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

const N           = 100
const N_DIS       = 32
const MAX_STEPS   = 1_000_000
const φ_ESCAPE    = 0.5
const σ_FLOOR     = 0.01

const ALPHA_TGRID = Dict(
    0.50 => collect(0.060:0.005:0.200),
    0.55 => collect(0.025:0.005:0.150),
    0.58 => collect(0.010:0.0025:0.100),
    0.60 => collect(0.005:0.0025:0.080),
    0.62 => collect(0.0025:0.0025:0.060),
)

const OUT_DIR    = @__DIR__
const STD_NORMAL = Normal()

function compute_K(α::Float64, N::Int)
    φ_c   = α + g_max
    half  = (N - 1) / 2
    log_comp = logccdf(Beta(half, half), (1.0 + φ_c) / 2.0)
    log_K = N * α + log_comp
    log_K > 40.0 && return -1
    return max(round(Int, exp(log_K)), 1)
end

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
        randn!(rng, u)
        unrm = norm(u)
        scale = sN * sqrt(max(0.0, 1.0 - φ_1μ * φ_1μ)) / unrm
        for i in 2:N
            P[k, i] = scale * u[i - 1]
        end
    end
    return P
end

@inline function log_Z(dots::Vector{Float64}, retr::Float64, log_bulk::Float64)
    max_val = max(retr, log_bulk)
    @inbounds @simd for v in dots
        max_val = max(max_val, v)
    end
    s = exp(retr - max_val) + exp(log_bulk - max_val)
    @inbounds @simd for v in dots
        s += exp(v - max_val)
    end
    return max_val + log(s)
end

function first_passage(α::Float64, T::Float64, N::Int, P::Matrix{Float64},
                       log_bulk::Float64, rng::AbstractRNG)
    K = size(P, 1)
    sN = sqrt(N)
    a = min(phi_eq(T), 0.999)
    x = zeros(N)
    x[1] = sN * a
    u = randn(rng, N - 1)
    u ./= norm(u)
    scale = sN * sqrt(max(0.0, 1.0 - a * a))
    @inbounds for i in 2:N
        x[i] = scale * u[i - 1]
    end

    dots = P * x
    retr = sN * x[1]
    logZ = log_Z(dots, retr, log_bulk)

    σ = max(σ_FLOOR, 2.0 * sqrt(T / N))
    η = Vector{Float64}(undef, N)
    x_new = Vector{Float64}(undef, N)
    dots_new = Vector{Float64}(undef, K)

    @inbounds for step in 1:MAX_STEPS
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
        end

        if x[1] / sN < φ_ESCAPE
            return step, true
        end
    end
    return MAX_STEPS, false
end

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
    csv_path = joinpath(OUT_DIR, @sprintf("escape_time_Ndim_N%d.csv", N))
    if fresh && isfile(csv_path); rm(csv_path); end
    done = load_existing(csv_path)
    @info "resume cells: $(length(done))"

    new_file = !isfile(csv_path)
    io = open(csv_path, "a")
    if new_file
        println(io,
            "# generator=escape_time_Ndim.jl  N=$N  N_DIS=$N_DIS  MAX_STEPS=$MAX_STEPS  φ_escape=$φ_ESCAPE  generated=$(Dates.now())")
        println(io, "alpha,T,disorder,K,tau,escaped")
        flush(io)
    end
    io_lock = ReentrantLock()

    # Group by (α, disorder) so we build patterns once
    αd_pairs = vec([(α, d) for α in collect(keys(ALPHA_TGRID)), d in 1:N_DIS])
    counter = Threads.Atomic{Int}(0)
    total = length(αd_pairs)

    @threads for (α, d) in αd_pairs
        if all((α, T, d) in done for T in ALPHA_TGRID[α])
            Threads.atomic_add!(counter, 1)
            continue
        end
        K = compute_K(α, N)
        if K < 0
            @warn "K too large at α=$α"
            Threads.atomic_add!(counter, 1)
            continue
        end
        rng_pat = MersenneTwister(hash((:Ndim_pat, N, α, d)))
        P = build_patterns(α, N, K, rng_pat)
        log_bulk = N * α + N * g_max

        for T in ALPHA_TGRID[α]
            (α, T, d) in done && continue
            rng_mc = MersenneTwister(hash((:Ndim_escape_mc, N, α, T, d)))
            τ, escaped = first_passage(α, T, N, P, log_bulk, rng_mc)
            lock(io_lock) do
                @printf(io, "%.3f,%.5f,%d,%d,%d,%d\n",
                        α, T, d, K, τ, escaped ? 1 : 0)
                flush(io)
            end
        end
        n = Threads.atomic_add!(counter, 1) + 1
        if n % max(1, total ÷ 20) == 0
            @info "$(n)/$(total)"
        end
    end
    close(io)
    @info "done → $csv_path"
end

main()

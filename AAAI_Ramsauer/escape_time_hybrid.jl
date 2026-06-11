#=
First-passage escape time τ from the LSE retrieval basin under the
hybrid 1D MC scheme.  Per (α, T, disorder, N) cell:
  start φ_1 = φ_eq(T)
  Metropolis on φ_1; record step at which φ_1 first crosses 0.5
  cap at MAX_STEPS (short budget — user will rerun longer later)

Output: escape_time_hybrid_N{N}.csv  (one CSV per N)
Resume on re-run; pass --fresh to overwrite.

Usage:
  julia -t auto escape_time_hybrid.jl
  julia -t auto escape_time_hybrid.jl --fresh
=#

using Base.Threads
using Distributions
using Random
using Printf
using Dates

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(φ_star) + φ_star

phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

# ───────── Grid & config ─────────
const N           = 100
const N_DIS       = 32
const MAX_STEPS   = 100_000
const φ_ESCAPE    = 0.5
const σ_FLOOR     = 0.01

# Per-α local T sweep around the empirical basin boundary (from the
# hybrid heatmap):  the basin loses stability at T ≈ T_boundary(α).
# We cover [T_lo, T_hi] with step ΔT chosen to give ~12 points each.
const ALPHA_TGRID = Dict(
    0.50 => collect(0.060:0.0025:0.100),   # boundary ≈ 0.085
    0.55 => collect(0.025:0.0025:0.060),   # boundary ≈ 0.045
    0.58 => collect(0.010:0.0025:0.040),   # boundary ≈ 0.025
    0.60 => collect(0.005:0.0025:0.035),   # boundary ≈ 0.015
    0.62 => collect(0.0025:0.0025:0.020),  # boundary ≈ 0.005
)

const OUT_DIR    = @__DIR__
const STD_NORMAL = Normal()

# ───────── K(α, N) and pattern sampling (spherical Beta tail) ─────────
function compute_K(α::Float64, N::Int)
    φ_c   = α + g_max
    half  = (N - 1) / 2
    log_comp = logccdf(Beta(half, half), (1.0 + φ_c) / 2.0)
    log_K = N * α + log_comp
    log_K > 40.0 && return -1
    return max(round(Int, exp(log_K)), 1)
end

function sample_overlaps(α::Float64, N::Int, K::Int, rng::AbstractRNG)
    φ_c  = α + g_max
    half = (N - 1) / 2
    tn = truncated(Beta(half, half), (1.0 + φ_c) / 2.0, 1.0)
    z = Vector{Float64}(undef, K)
    @inbounds for k in 1:K
        z[k] = 2.0 * rand(rng, tn) - 1.0
    end
    sort!(z; rev = true)
    return z
end

# ───────── log Z(φ_1) (same as hybrid_1d_LSE.jl) ─────────
@inline function log_Z_hybrid(φ_1::Float64, z::Vector{Float64}, α::Float64, N::Int)
    omφ² = 1.0 - φ_1 * φ_1
    retrieved = N * φ_1
    bulk = N * (α + g_max)

    max_val = max(retrieved, bulk)
    if !isempty(z)
        zmax = φ_1 >= 0 ? z[1] : z[end]
        kept_max = N * (φ_1 * zmax + 0.5 * (1.0 - zmax * zmax) * omφ²)
        max_val = max(max_val, kept_max)
    end

    s = exp(retrieved - max_val) + exp(bulk - max_val)
    @inbounds @simd for k in eachindex(z)
        zk = z[k]
        term = N * (φ_1 * zk + 0.5 * (1.0 - zk * zk) * omφ²)
        s += exp(term - max_val)
    end
    return max_val + log(s)
end

# ───────── First-passage τ ─────────
function first_passage(α::Float64, T::Float64, N::Int,
                       z::Vector{Float64}, rng::AbstractRNG)
    φ_1  = min(phi_eq(T), 0.999)
    logZ = log_Z_hybrid(φ_1, z, α, N)
    σ = max(σ_FLOOR, 2.0 * sqrt(T / N))

    @inbounds for step in 1:MAX_STEPS
        φ_new = φ_1 + σ * randn(rng)
        if -0.999 < φ_new < 0.999
            logZ_new = log_Z_hybrid(φ_new, z, α, N)
            log_acc  = (logZ_new - logZ) / T +
                       0.5 * N * (log1p(-φ_new * φ_new) - log1p(-φ_1 * φ_1))
            if log_acc >= 0.0 || log(rand(rng)) < log_acc
                φ_1  = φ_new
                logZ = logZ_new
            end
        end
        if φ_1 < φ_ESCAPE
            return step, true
        end
    end
    return MAX_STEPS, false
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
    csv_path = joinpath(OUT_DIR, @sprintf("escape_time_hybrid_N%d.csv", N))

    if fresh && isfile(csv_path)
        rm(csv_path)
    end
    done = load_existing(csv_path)
    @info "resume cells: $(length(done))"

    new_file = !isfile(csv_path)
    io = open(csv_path, "a")
    if new_file
        println(io,
            "# generator=escape_time_hybrid.jl  N=$N  N_DIS=$N_DIS  MAX_STEPS=$MAX_STEPS  φ_escape=$φ_ESCAPE  generated=$(Dates.now())")
        println(io, "alpha,T,disorder,K,tau,escaped")
        flush(io)
    end
    io_lock = ReentrantLock()

    tasks = Tuple{Float64,Float64,Int}[]
    for (α, Ts) in pairs(ALPHA_TGRID), T in Ts, d in 1:N_DIS
        push!(tasks, (α, T, d))
    end

    counter = Threads.Atomic{Int}(0)
    total = length(tasks)
    @threads for task in tasks
        α, T, d = task
        (α, T, d) in done && (Threads.atomic_add!(counter, 1); continue)
        K = compute_K(α, N)
        if K < 0
            @warn "K too large at α=$α — skip"
            Threads.atomic_add!(counter, 1)
            continue
        end
        rng_pat = MersenneTwister(hash((:hybrid_pat, N, α, d)))
        z = sample_overlaps(α, N, K, rng_pat)
        rng_mc = MersenneTwister(hash((:escape_mc, N, α, T, d)))
        τ, escaped = first_passage(α, T, N, z, rng_mc)
        lock(io_lock) do
            @printf(io, "%.3f,%.5f,%d,%d,%d,%d\n",
                    α, T, d, K, τ, escaped ? 1 : 0)
            flush(io)
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

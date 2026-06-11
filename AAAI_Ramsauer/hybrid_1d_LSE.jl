#=
1D hybrid MC for LSE basin escape near the saddle-dominance tip.

Per (α, T, disorder, N) cell:
  cusp φ_c = α + g_max
  K(α, N) = round(M · (1 − Φ(φ_c √N)))                # patterns with overlap above cusp
  Sample K overlaps z_μ from truncated Gaussian z > φ_c (z ~ N(0, 1/N))

  Energy (per N):
    H(φ_1)/N = −(1/N) log Z(φ_1)
    Z(φ_1)   = exp(N φ_1)
               + Σ_μ exp(N z_μ φ_1)                   (kept tail)
               + M · exp(N φ_1²/2) · Φ((φ_c − φ_1) √N) (Gaussian bulk, z < φ_c)

  Free energy (per N):
    F(φ_1)/N = H(φ_1)/N − (T/2) log(1 − φ_1²)

  Boltzmann target on φ_1: exp(−F·N/T) = exp(−H·N/T) · (1−φ_1²)^{N/2}
  Metropolis: accept with prob min(1, exp((log Z' − log Z)/T + (N/2) log((1−φ'²)/(1−φ²))))

Grid: α ∈ {0.50, 0.51, …, 0.62},  T ∈ {0.0025, 0.0075, …, 0.0975},  N ∈ {50, 100, 150},  N_DIS = 32.

Output: hybrid_1d_LSE_N{50,100,150}.csv next to this script.
Resume on re-run; pass --fresh to overwrite.

Usage:
  julia -t auto hybrid_1d_LSE.jl
  julia -t auto hybrid_1d_LSE.jl --fresh
=#

using Base.Threads
using Distributions
using Random
using Printf
using Dates

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(φ_star) + φ_star          # ≈ 0.3774
# Spherical-saddle bulk value g_max = max_z [z + (1/2) log(1-z²)] at z = φ_star.
# This is the per-pattern contribution from random patterns to log Z (independent
# of φ_1 by rotational symmetry of the orthogonal sector).

phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

# ───────── Grid & config ─────────
const ALPHA_GRID = collect(0.50:0.01:0.62)         # 13 values
const T_GRID     = collect(0.0025:0.005:0.0975)    # 20 values
const N_LIST     = [100]
const N_DIS      = 32

const N_EQ       = 4_000
const N_SAMP     = 16_000
const σ_FLOOR    = 0.01

const OUT_DIR    = @__DIR__
const STD_NORMAL = Normal()

# ───────── K(α, N) via cusp rule — spherical pattern-overlap density ─────────
# c_μ = ⟨ξ^μ, ξ^1⟩/N has density ρ(c) ∝ (1 − c²)^{(N-3)/2}, equivalently
# (1+c)/2 ~ Beta((N-1)/2, (N-1)/2).
function compute_K(α::Float64, N::Int)
    φ_c   = α + g_max
    half  = (N - 1) / 2
    dist  = Beta(half, half)
    log_comp = logccdf(dist, (1.0 + φ_c) / 2.0)     # log P(c > φ_c)
    log_K = N * α + log_comp
    log_K > 40.0 && return -1
    return max(round(Int, exp(log_K)), 1)
end

# ───────── Sample K overlap values c > φ_c from spherical density ─────────
function sample_overlaps(α::Float64, N::Int, K::Int, rng::AbstractRNG)
    φ_c   = α + g_max
    half  = (N - 1) / 2
    tn = truncated(Beta(half, half), (1.0 + φ_c) / 2.0, 1.0)
    z = Vector{Float64}(undef, K)
    @inbounds for k in 1:K
        z[k] = 2.0 * rand(rng, tn) - 1.0
    end
    sort!(z; rev = true)                            # descending → z[1] is max
    return z
end

# ───────── log Z(φ_1) ─────────
# Bulk term: M · exp(N · g_max), independent of φ_1.
#   Rationale: for random patterns on the sphere, the marginal of ⟨ξ^μ, x⟩/N is
#   the spherical overlap density (1−r²)^{(N-3)/2} regardless of x's direction.
#   Its saddle with weight exp(Nr) is at r=φ_star with value g_max ≈ 0.378.
# Kept tail (patterns with c_μ > φ_c): explicit sum with per-pattern exponent
#   g(φ_1, c) = φ_1·c + (1 − c²)(1 − φ_1²)/2
#   which is the Gaussian saddle over the orthogonal-sector overlap. For
#   c > φ_c (close to 1), (1−c²) is small so the Gaussian saddle is accurate.
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

# ───────── 1D Metropolis ─────────
function run_mc(α::Float64, T::Float64, N::Int, z::Vector{Float64}, rng::AbstractRNG)
    φ_1 = min(phi_eq(T), 0.999)
    logZ = log_Z_hybrid(φ_1, z, α, N)

    σ = max(σ_FLOOR, 2.0 * sqrt(T / N))

    sum_φ  = 0.0
    sum_φ2 = 0.0
    n_acc  = 0
    n_steps = N_EQ + N_SAMP
    @inbounds for step in 1:n_steps
        φ_new = φ_1 + σ * randn(rng)
        if -0.999 < φ_new < 0.999
            logZ_new = log_Z_hybrid(φ_new, z, α, N)
            log_acc  = (logZ_new - logZ) / T +
                       0.5 * N * (log1p(-φ_new * φ_new) - log1p(-φ_1 * φ_1))
            if log_acc >= 0.0 || log(rand(rng)) < log_acc
                φ_1   = φ_new
                logZ  = logZ_new
                n_acc += 1
            end
        end
        if step > N_EQ
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

# ───────── Per-N driver ─────────
function run_for_N(N::Int; fresh::Bool = false)
    csv_path = joinpath(OUT_DIR, @sprintf("hybrid_1d_LSE_N%d.csv", N))

    if fresh && isfile(csv_path)
        rm(csv_path)
    end
    done = load_existing(csv_path)
    @info "N=$N  resume cells: $(length(done))"

    new_file = !isfile(csv_path)
    io = open(csv_path, "a")
    if new_file
        println(io,
            "# generator=hybrid_1d_LSE.jl  N=$N  N_DIS=$N_DIS  N_EQ=$N_EQ  N_SAMP=$N_SAMP  σ_floor=$σ_FLOOR  generated=$(Dates.now())")
        println(io, "alpha,T,disorder,K,phi_mean,phi_var,acc_rate")
        flush(io)
    end
    io_lock = ReentrantLock()

    tasks = vec([(d, α) for d in 1:N_DIS, α in ALPHA_GRID])

    counter = Threads.Atomic{Int}(0)
    total   = length(tasks)
    @threads for task in tasks
        d, α = task
        # skip whole (α, d) if every T already done
        if all((α, T, d) in done for T in T_GRID)
            Threads.atomic_add!(counter, 1)
            continue
        end
        K = compute_K(α, N)
        if K < 0
            @warn "N=$N α=$α  K too large; skipping"
            Threads.atomic_add!(counter, 1)
            continue
        end
        rng_pat = MersenneTwister(hash((:hybrid_pat, N, α, d)))
        z = sample_overlaps(α, N, K, rng_pat)

        for T in T_GRID
            (α, T, d) in done && continue
            rng_mc = MersenneTwister(hash((:hybrid_mc, N, α, T, d)))
            mφ, vφ, acc = run_mc(α, T, N, z, rng_mc)
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
    @info "N=$N done → $csv_path"
end

# ───────── Main ─────────
fresh = "--fresh" in ARGS
for N in N_LIST
    run_for_N(N; fresh = fresh)
end

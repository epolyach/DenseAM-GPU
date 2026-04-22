#=
Finite-Size Scaling for LSE — CPU multi-threaded
────────────────────────────────────────────────────────────────────────
Goal: At fixed α, vary N (by varying M = e^{αN}) and measure:
  1. ⟨φ⟩ vs T for each N → where does the boundary move?
  2. Per-trial φ distribution → sharp or gradual transition?
  3. φ_{1,max} per disorder sample → does extreme overlap drive the transition?
  4. φ_max_other → is non-retrieval due to interference or escape?

Usage:
  julia -t 10 fss_LSE_cpu.jl        # use 10 threads
  julia -t auto fss_LSE_cpu.jl      # auto-detect cores
────────────────────────────────────────────────────────────────────────
=#

using Random
using Statistics
using LinearAlgebra
using Printf
using Base.Threads

# ──────────────── Parameters ────────────────
const F = Float64
const betanet = 1.0

const ALPHA = 0.20                    # fixed α for scaling study
const N_VALUES = [20, 25, 30, 35]     # N values (M = exp(α×N))
const N_EQ   = 2^14                   # 16384 equilibration steps
const N_SAMP = 2^12                   # 4096 sampling steps
const N_DIS  = 200                    # disorder samples per (N, T)
const PHI_MIN = 0.75
const PHI_MAX = 1.0

# T grid: exponentially concentrated at low T
const T_VEC = let dw = 0.1; fi = [exp(dw*i) - 1 for i in 1:40]; fi ./ maximum(fi); end

# ──────────────── LSE Energy (CPU, single chain) ────────────────
function compute_energy(x::Vector{F}, patterns::Matrix{F}, N::Int)
    # H = -(1/βnet) ln Σ exp(βnet x·ξ^μ)
    # Use log-sum-exp trick for stability
    overlaps = patterns' * x  # M-vector
    shifted = @. -betanet * (N - overlaps)  # = βnet(x·ξ^μ - N)
    m = maximum(shifted)
    return -(m + log(sum(exp.(shifted .- m)))) / betanet
end

# ──────────────── MC Step ────────────────
function mc_step!(x::Vector{F}, E::Ref{F}, patterns::Matrix{F},
                  N::Int, β::F, σ::F, xp::Vector{F})
    randn!(xp)
    @. xp = x + σ * xp
    xp .*= sqrt(F(N)) / norm(xp)  # project onto sphere

    Ep = compute_energy(xp, patterns, N)
    if rand() < exp(-β * (Ep - E[]))
        x .= xp
        E[] = Ep
    end
end

# ──────────────── Initialize near target ────────────────
function initialize_near(target::Vector{F}, N::Int)
    phi_init = PHI_MIN + (PHI_MAX - PHI_MIN) * rand()
    x_perp = randn(N)
    x_perp .-= dot(target, x_perp) / N .* target
    x_perp ./= norm(x_perp)
    return phi_init .* target .+ sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
end

# ──────────────── Run one trial ────────────────
function run_trial(N::Int, M::Int, T::F, seed::Int)
    rng = MersenneTwister(seed)

    # Generate patterns
    patterns = randn(rng, N, M)
    for j in 1:M
        c = @view patterns[:, j]
        c .*= sqrt(F(N)) / norm(c)
    end
    target = patterns[:, 1]

    # Compute φ_{1,max}: max overlap of target with spurious patterns
    phi_1max = -Inf
    for j in 2:M
        ov = dot(target, patterns[:, j]) / N
        phi_1max = max(phi_1max, ov)
    end

    # Initialize two replicas
    Random.seed!(rng, seed + 1000000)
    xa = initialize_near(target, N)
    Random.seed!(rng, seed + 2000000)
    xb = initialize_near(target, N)

    β = 1 / T
    σ = 2.4 * T / sqrt(F(N))
    xp = zeros(N)

    Ea = Ref(compute_energy(xa, patterns, N))
    Eb = Ref(compute_energy(xb, patterns, N))

    # Equilibrate
    for _ in 1:N_EQ
        mc_step!(xa, Ea, patterns, N, β, σ, xp)
        mc_step!(xb, Eb, patterns, N, β, σ, xp)
    end

    # Sample
    phi_a_sum = 0.0; phi_b_sum = 0.0; q_sum = 0.0
    phimax_a_sum = 0.0
    for _ in 1:N_SAMP
        mc_step!(xa, Ea, patterns, N, β, σ, xp)
        mc_step!(xb, Eb, patterns, N, β, σ, xp)

        phi_a_sum += dot(target, xa) / N
        phi_b_sum += dot(target, xb) / N
        q_sum += dot(xa, xb) / N

        # φ_max_other for replica A
        mx = -Inf
        for j in 2:M
            mx = max(mx, dot(patterns[:, j], xa) / N)
        end
        phimax_a_sum += mx
    end

    phi_a = phi_a_sum / N_SAMP
    phi_b = phi_b_sum / N_SAMP
    q_ea  = q_sum / N_SAMP
    phimax = phimax_a_sum / N_SAMP

    return (phi_a=phi_a, phi_b=phi_b, q_ea=q_ea, phimax=phimax, phi_1max=phi_1max)
end

# ──────────────── Main ────────────────
function main()
    n_T = length(T_VEC)
    total_tasks = length(N_VALUES) * n_T * N_DIS

    println("=" ^ 70)
    println("Finite-Size Scaling for LSE (CPU, $(nthreads()) threads)")
    println("  α = $ALPHA")
    println("  N values: $N_VALUES")
    println("  T values: $n_T (exponential grid, T_max = $(round(T_VEC[end], digits=3)))")
    println("  n_dis = $N_DIS, N_eq = $N_EQ, N_samp = $N_SAMP")
    println("  Total trials: $total_tasks")
    println("=" ^ 70)

    outfile = "fss_LSE_alpha$(ALPHA).csv"
    open(outfile, "w") do f
        write(f, "alpha,N,M,T,disorder,phi_a,phi_b,q_ea,phi_max_other,phi_1max\n")
    end

    for N in N_VALUES
        M = round(Int, exp(ALPHA * N))
        @printf("\nN = %d, M = %d:\n", N, M)

        for (iT, T) in enumerate(T_VEC)
            t_start = time()

            # Run N_DIS trials in parallel
            results = Vector{NamedTuple}(undef, N_DIS)
            @threads for d in 1:N_DIS
                seed = N * 100000 + iT * 1000 + d
                results[d] = run_trial(N, M, T, seed)
            end

            # Write results
            open(outfile, "a") do f
                for d in 1:N_DIS
                    r = results[d]
                    @printf(f, "%.2f,%d,%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                            ALPHA, N, M, T, d,
                            r.phi_a, r.phi_b, r.q_ea, r.phimax, r.phi_1max)
                end
            end

            phi_mean = mean([r.phi_a for r in results] ∪ [r.phi_b for r in results])
            phi1max_mean = mean([r.phi_1max for r in results])
            elapsed = time() - t_start
            @printf("  T=%.4f: ⟨φ⟩=%.3f, ⟨φ_{1,max}⟩=%.3f (%.1f s)\n",
                    T, phi_mean, phi1max_mean, elapsed)
        end
    end

    println("\nSaved: $outfile")
end

main()

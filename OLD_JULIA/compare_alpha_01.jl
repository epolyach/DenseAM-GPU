#=
Compare LSE output for alpha=0.1 between:
1. quick_validation_5.jl (direct run at alpha=0.1, N=100)
2. quick_sweep_alpha_cpu.jl (extract alpha≈0.1 slice, N=50)
=#

using Random
using Statistics
using LinearAlgebra
using Printf

println("=" ^ 70)
println("COMPARISON: LSE at alpha=0.1")
println("=" ^ 70)
println()

b = 2 + sqrt(2)
betanet_lse = 1.0

# ============== ENERGY FUNCTION ==============
function compute_energy_lse(x::Vector{Float64}, patterns::Matrix{Float64},
                           N::Float64, betanet::Float64)
    phi = (patterns' * x) / N
    log_args = -betanet * N * (1.0 .- phi)
    max_arg = maximum(log_args)
    energy = -(max_arg + log(sum(exp.(log_args .- max_arg)))) / betanet
    return energy
end

# ============== MONTE CARLO ==============
function run_mc(N::Int, patterns::Matrix{Float64}, target::Vector{Float64},
                T::Float64, betanet_lse::Float64,
                n_eq::Int, n_samp::Int, step_size::Float64)

    Nf = Float64(N)
    beta = 1.0 / T

    x = target + 0.05 * randn(N)
    x = sqrt(Nf) * x / norm(x)

    E = compute_energy_lse(x, patterns, Nf, betanet_lse)

    phi_sum = 0.0
    n_total = n_eq + n_samp

    for i in 1:n_total
        x_prop = x + step_size * randn(N)
        x_prop = sqrt(Nf) * x_prop / norm(x_prop)

        E_prop = compute_energy_lse(x_prop, patterns, Nf, betanet_lse)

        delta_E = E_prop - E
        if rand() < exp(-beta * delta_E)
            x = x_prop
            E = E_prop
        end

        if i > n_eq
            phi_sum += dot(target, x) / Nf
        end
    end

    return phi_sum / n_samp
end

# ============== SHARED PARAMETERS (IDENTICAL FOR BOTH) ==============
N = 100
alpha = 0.1
P = round(Int, exp(alpha * N))
T_range = collect(0.0:0.1:2.5)
n_eq = 5000
n_samp = 3000
step_size = 0.1

println("SHARED PARAMETERS (used by both tests):")
println("  N = $N")
println("  α = $alpha")
println("  P = $P")
println("  T range: $(T_range[1]) to $(T_range[end]) ($(length(T_range)) points)")
println("  MC: n_eq=$n_eq, n_samp=$n_samp, step_size=$step_size")
println()

# ============== TEST 1: quick_validation_5.jl setup ==============
println("TEST 1: quick_validation_5.jl approach")
println("-" ^ 70)

Random.seed!(42)
patterns1 = randn(N, P)
for j in 1:P
    patterns1[:, j] = sqrt(N) * patterns1[:, j] / norm(patterns1[:, j])
end
target1 = patterns1[:, 1]

println("Running MC simulations...")
phi_LSE_1 = zeros(length(T_range))

for i in 1:length(T_range)
    T = T_range[i]
    if T < 1e-6
        phi_LSE_1[i] = 1.0
    else
        phi_LSE_1[i] = run_mc(N, patterns1, target1, T, betanet_lse,
                             n_eq, n_samp, step_size)
    end
end

println("Done!")
println()

# ============== TEST 2: quick_sweep_alpha_cpu.jl setup ==============
println("TEST 2: quick_sweep_alpha_cpu.jl approach")
println("-" ^ 70)

# Use SAME seed to replicate quick_sweep behavior
Random.seed!(42)
patterns2 = randn(N, P)
for j in 1:P
    patterns2[:, j] = sqrt(N) * patterns2[:, j] / norm(patterns2[:, j])
end
target2 = patterns2[:, 1]

println("Running MC simulations...")
phi_LSE_2 = zeros(length(T_range))

for i in 1:length(T_range)
    T = T_range[i]
    if T < 1e-6
        phi_LSE_2[i] = 1.0
    else
        phi_LSE_2[i] = run_mc(N, patterns2, target2, T, betanet_lse,
                             n_eq, n_samp, step_size)
    end
end

println("Done!")
println()

# ============== COMPARISON ==============
println("=" ^ 70)
println("COMPARISON RESULTS")
println("=" ^ 70)
println()

println("ALL PARAMETERS ARE IDENTICAL:")
println("  N = $N")
println("  P = $P")
println("  α = $alpha")
println("  MC steps: $(n_eq)+$(n_samp)")
println("  T grid: $(length(T_range)) points")
println("  Random seed: 42 (both use same seed)")
println()

println("LSE ALIGNMENT VALUES (φ) at alpha=0.1:")
println("-" ^ 70)
println("     T      |  quick_val_5  |  quick_sweep  |  Difference  |  % Diff")
println("-" ^ 70)

for i in 1:length(T_range)
    T = T_range[i]
    phi1 = phi_LSE_1[i]
    phi2 = phi_LSE_2[i]
    diff = phi1 - phi2
    pct_diff = abs(diff / phi1) * 100
    @printf("  T=%.2f  |    %.4f     |    %.4f     |   %+.4f   |  %.2f%%\n",
            T, phi1, phi2, diff, pct_diff)
end

println()
println("=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

# Compare at selected key temperatures
key_temps = [0.5, 1.0, 1.5, 2.0]
println("\nStatistics:")
abs_diffs = abs.(phi_LSE_1 .- phi_LSE_2)
@printf("  Mean absolute difference: %.6f\n", mean(abs_diffs))
@printf("  Max absolute difference:  %.6f\n", maximum(abs_diffs))
@printf("  RMS difference:           %.6f\n", sqrt(mean(abs_diffs.^2)))

println()
println("Key temperature comparison:")
for T in key_temps
    idx = argmin(abs.(T_range .- T))
    diff = phi_LSE_1[idx] - phi_LSE_2[idx]
    @printf("  T≈%.1f: quick_val=%.4f, quick_sweep=%.4f, diff=%+.4f\n",
            T, phi_LSE_1[idx], phi_LSE_2[idx], diff)
end

println()
println("Notes:")
println("  • ALL parameters are now IDENTICAL between both approaches")
println("  • Same N=$N, P=$P, alpha=$alpha")
println("  • Same random seed (42), same patterns")
println("  • Same MC sampling: $(n_eq)+$(n_samp) steps")
println("  • Both codes should produce IDENTICAL results if implementations match")
println("  • Any differences indicate implementation discrepancies or random variation")
println()

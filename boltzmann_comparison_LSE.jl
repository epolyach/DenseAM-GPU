using CSV
using DataFrames
using Plots
using Printf
using Statistics

# LSE parameters
betanet = 1.0

alpha = 0.10

println("="^70)
println("Boltzmann Statistics vs Observed Data for LSE, α = $alpha")
println("="^70)

# Read data
data = CSV.read("basin_stab_LSE_v6.csv", DataFrame)
row_idx = findfirst(data.alpha .== alpha)
row_data = data[row_idx, :]

# Extract T values and φ values
T_cols = names(data)[2:end]
T_vals = [parse(Float64, replace(col, "T" => "")) for col in T_cols]
phi_obs = Vector(row_data[2:end])

# Calculate N and P using the same power-law scheme as v3
MIN_PAT = 20000
MAX_PAT = 500000
n_alpha = 55
ind = 10
alpha_vec = collect(0.01:0.01:0.55)
n_patterns_vec = range(MIN_PAT^(1/ind), MAX_PAT^(1/ind), length=n_alpha) .^ ind

alpha_idx = findfirst(alpha_vec .≈ alpha)
P = round(Int, n_patterns_vec[alpha_idx])
N = round(Int, log(P) / alpha)

println("\nFor α = $alpha:")
println("  P = $P patterns")
println("  N = $N dimensions")

# LSE energy density: u(φ) = 1 - φ
function energy_per_N(phi)
    return 1.0 - phi
end

# Density of states on N-dimensional sphere (log-space)
function log_rho(phi, N)
    if abs(phi) >= 1.0 - 1e-12
        return -Inf
    else
        return ((N-2)/2) * log(1 - phi^2)
    end
end

# Log of Boltzmann weight
function log_boltzmann(phi, N, T)
    E = energy_per_N(phi) * N
    return -E / T
end

# Log of thermal probability (unnormalized)
function log_prob(phi, N, T)
    return log_rho(phi, N) + log_boltzmann(phi, N, T)
end

# Numerical integration using log-sum-exp trick for stability
function thermal_average_phi_stable(N, T)
    n_points = 5000
    # LSE: integrate over [-1, 1] (no hard wall)
    phi_grid = range(-1.0 + 1e-10, 1.0 - 1e-10, length=n_points)
    dphi = phi_grid[2] - phi_grid[1]

    # Calculate log probabilities
    log_probs = [log_prob(phi, N, T) for phi in phi_grid]

    # Log-sum-exp trick
    log_prob_max = maximum(log_probs)

    Z_terms = [exp(lp - log_prob_max) for lp in log_probs]
    Z = sum(Z_terms) * dphi

    phi_times_prob = [phi_grid[i] * exp(log_probs[i] - log_prob_max) for i in 1:n_points]
    phi_avg = sum(phi_times_prob) * dphi / Z

    return phi_avg
end

# Analytical saddle-point result (Eq. 33): φ_eq(T) = ½[-T + √(T² + 4)]
function phi_analytical(T)
    return 0.5 * (-T + sqrt(T^2 + 4))
end

println("\n" * "="^70)
println("Computing Theoretical Curves (Boltzmann Integration + Analytical)")
println("="^70)

phi_theory = Float64[]
phi_analyt = Float64[]
for (i, T) in enumerate(T_vals)
    phi_th = thermal_average_phi_stable(N, T)
    phi_an = phi_analytical(T)
    push!(phi_theory, phi_th)
    push!(phi_analyt, phi_an)

    if i % 10 == 0 || i == 1 || i == length(T_vals)
        println(@sprintf("T = %.4f: ⟨φ⟩_Boltz = %.4f, φ_analyt = %.4f, φ_obs = %.4f, Δ(obs-Boltz) = %+.4f",
                         T, phi_th, phi_an, phi_obs[i], phi_obs[i] - phi_th))
    end
end

println("\n" * "="^70)
println("Statistical Analysis")
println("="^70)

differences = phi_obs .- phi_theory
rms_diff = sqrt(mean(differences.^2))
max_diff = maximum(abs.(differences))
mean_diff = mean(differences)

println(@sprintf("\nBoltzmann integral vs MC:"))
println(@sprintf("  RMS difference: %.4f", rms_diff))
println(@sprintf("  Max |difference|: %.4f", max_diff))
println(@sprintf("  Mean difference: %+.4f", mean_diff))
println(@sprintf("  Relative RMS: %.2f%%", 100 * rms_diff / mean(phi_obs)))

diff_analyt = phi_obs .- phi_analyt
rms_analyt = sqrt(mean(diff_analyt.^2))
println(@sprintf("\nAnalytical (Eq.33) vs MC:"))
println(@sprintf("  RMS difference: %.4f", rms_analyt))

diff_boltz_analyt = phi_theory .- phi_analyt
rms_boltz_analyt = sqrt(mean(diff_boltz_analyt.^2))
println(@sprintf("\nBoltzmann integral vs Analytical (finite-N correction):"))
println(@sprintf("  RMS difference: %.4f", rms_boltz_analyt))

# Create plot
println("\n" * "="^70)
println("Creating Plot")
println("="^70)

p = plot(size=(800, 600), dpi=150)

# Observed data
plot!(p, T_vals, phi_obs,
      label="Observed (MC data)",
      linewidth=3,
      marker=:circle,
      markersize=4,
      color=:blue,
      markerstrokewidth=0)

# Theoretical prediction (Boltzmann integral)
plot!(p, T_vals, phi_theory,
      label="Theory (Boltzmann integral, N=$N)",
      linewidth=3,
      linestyle=:dash,
      color=:red)

# Analytical saddle-point (N → ∞)
plot!(p, T_vals, phi_analyt,
      label="Analytical (Eq.33, N→∞)",
      linewidth=2,
      linestyle=:dashdot,
      color=:green)

# Formatting
xlabel!(p, "Temperature T")
ylabel!(p, "Alignment φ")
title!(p, "LSE: Boltzmann Statistics vs Observed Data (α = $alpha, N = $N)")

# Add text annotation
annotate!(p, 1.5, 0.85,
         text("P = $P\nN = $N\nRMS diff = $(round(rms_diff, digits=4))",
              :left, 10))

# Save plot
alpha_str = replace(@sprintf("%.2f", alpha), "." => "")
fname_png = "boltzmann_comparison_LSE_alpha$(alpha_str).png"
savefig(p, fname_png)
println("✓ Plot saved: $fname_png")

# Save data to CSV
println("\n" * "="^70)
println("Saving Data")
println("="^70)

df_out = DataFrame(
    T = T_vals,
    phi_observed = phi_obs,
    phi_theory = phi_theory,
    phi_analytical = phi_analyt,
    difference = differences
)

fname_csv = "boltzmann_comparison_LSE_alpha$(alpha_str).csv"
CSV.write(fname_csv, df_out)
println("✓ Data saved: $fname_csv")

println("\n" * "="^70)
println("Conclusion")
println("="^70)

if rms_diff < 0.02
    println("\n✓ EXCELLENT AGREEMENT!")
    println("  The observed data matches Boltzmann statistics within $(round(100*rms_diff/mean(phi_obs), digits=2))%")
    println("  This confirms that at α = $alpha:")
    println("    • System is in thermal equilibrium")
    println("    • No glassy effects or metastability")
    println("    • Pure statistical mechanics in confining potential")
elseif rms_diff < 0.05
    println("\n✓ GOOD AGREEMENT")
    println("  The observed data reasonably matches Boltzmann statistics")
    println("  Small deviations (~$(round(100*rms_diff/mean(phi_obs), digits=1))%) may be due to:")
    println("    • Finite sampling")
    println("    • Finite N effects")
    println("    • Approximations in density of states")
else
    println("\n✗ SIGNIFICANT DEVIATION")
    println("  The observed data deviates from Boltzmann prediction")
    println("  This suggests non-equilibrium effects or model issues")
end

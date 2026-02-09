using CSV
using DataFrames
using Plots
using Printf
using Statistics

# LSR parameters
b = 2 + sqrt(2)  # ≈ 3.414
phi_c = (b - 1) / b  # Critical threshold ≈ 0.707

println("="^70)
println("Boltzmann Statistics vs Observed Data for α = 0.1")
println("="^70)

# Read data
# data = CSV.read("lsr_longeq.csv", DataFrame)
data = CSV.read("basin_stab_LSR_v3.csv", DataFrame)

# Extract α = 0.1 data
alpha = 0.1
row_idx = findfirst(data.alpha .== alpha)
row_data = data[row_idx, :]

# Extract T values and φ values
T_cols = names(data)[2:end]
T_vals = [parse(Float64, replace(col, "T" => "")) for col in T_cols]
phi_obs = Vector(row_data[2:end])

# Calculate N and P for α = 0.1
P = round(Int, 500 + alpha * (20000 - 500) / 0.54)
N = round(Int, P / alpha)

println("\nFor α = $alpha:")
println("  P = $P patterns")
println("  N = $N dimensions")
println("  Hard wall at φ_c = $(round(phi_c, digits=4))")

# Energy function (per particle)
function energy_per_N(phi)
    if phi <= phi_c + 1e-12
        return 10.0  # Large but finite to avoid numerical issues
    else
        return -(1/b) * log(b * (phi - phi_c))
    end
end

# Density of states on N-dimensional sphere
# For large N, work in log-space
function log_rho(phi, N)
    if abs(phi) >= 1.0 - 1e-12
        return -Inf
    else
        return ((N-3)/2) * log(1 - phi^2)
    end
end

# Log of Boltzmann weight
function log_boltzmann(phi, N, T)
    E = energy_per_N(phi) * N
    return -E / T
end

# Log of thermal probability (unnormalized)
function log_prob(phi, N, T)
    lrho = log_rho(phi, N)
    lboltz = log_boltzmann(phi, N, T)
    return lrho + lboltz
end

# Numerical integration using log-sum-exp trick for stability
function thermal_average_phi_stable(N, T)
    # Integration grid
    n_points = 5000
    phi_grid = range(phi_c + 1e-8, 1.0 - 1e-10, length=n_points)
    dphi = phi_grid[2] - phi_grid[1]

    # Calculate log probabilities
    log_probs = [log_prob(phi, N, T) for phi in phi_grid]

    # Find maximum for numerical stability (log-sum-exp trick)
    log_prob_max = maximum(log_probs)

    # Calculate normalization (in log space)
    Z_terms = [exp(lp - log_prob_max) for lp in log_probs]
    Z = sum(Z_terms) * dphi

    # Calculate average
    phi_times_prob = [phi_grid[i] * exp(log_probs[i] - log_prob_max) for i in 1:n_points]
    phi_avg = sum(phi_times_prob) * dphi / Z

    return phi_avg
end

println("\n" * "="^70)
println("Computing Theoretical Curve (Boltzmann Integration)")
println("="^70)

# Calculate theoretical predictions
phi_theory = Float64[]
for (i, T) in enumerate(T_vals)
    phi_th = thermal_average_phi_stable(N, T)
    push!(phi_theory, phi_th)

    if i % 10 == 0 || i == 1 || i == length(T_vals)
        println(@sprintf("T = %.4f: ⟨φ⟩_theory = %.4f, φ_obs = %.4f, Δ = %+.4f",
                         T, phi_th, phi_obs[i], phi_obs[i] - phi_th))
    end
end

println("\n" * "="^70)
println("Statistical Analysis")
println("="^70)

# Calculate RMS difference
differences = phi_obs .- phi_theory
rms_diff = sqrt(mean(differences.^2))
max_diff = maximum(abs.(differences))
mean_diff = mean(differences)

println(@sprintf("\nRMS difference: %.4f", rms_diff))
println(@sprintf("Max |difference|: %.4f", max_diff))
println(@sprintf("Mean difference: %+.4f", mean_diff))
println(@sprintf("Relative RMS: %.2f%%", 100 * rms_diff / mean(phi_obs)))

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

# Theoretical prediction
plot!(p, T_vals, phi_theory,
      label="Theory (Boltzmann integral)",
      linewidth=3,
      linestyle=:dash,
      color=:red)

# Formatting
xlabel!(p, "Temperature T")
ylabel!(p, "Alignment φ")
title!(p, "LSR: Boltzmann Statistics vs Observed Data (α = $alpha, N = $N)")

# Add horizontal line at hard wall
hline!(p, [phi_c],
       label="Hard wall φ_c = $(round(phi_c, digits=3))",
       linestyle=:dot,
       linewidth=2,
       color=:black)

# Add text annotation
annotate!(p, 1.5, 0.95,
         text("P = $P\nN = $N\nRMS diff = $(round(rms_diff, digits=4))",
              :left, 10))

# Save plot
savefig(p, "boltzmann_comparison_alpha01.png")
println("✓ Plot saved: boltzmann_comparison_alpha01.png")

# Save data to CSV
println("\n" * "="^70)
println("Saving Data")
println("="^70)

df_out = DataFrame(
    T = T_vals,
    phi_observed = phi_obs,
    phi_theory = phi_theory,
    difference = differences
)

CSV.write("boltzmann_comparison_alpha01.csv", df_out)
println("✓ Data saved: boltzmann_comparison_alpha01.csv")

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
    println("    • Finite sampling (256 trials)")
    println("    • Finite N effects")
    println("    • Approximations in density of states")
else
    println("\n✗ SIGNIFICANT DEVIATION")
    println("  The observed data deviates from Boltzmann prediction")
    println("  This suggests non-equilibrium effects or model issues")
end

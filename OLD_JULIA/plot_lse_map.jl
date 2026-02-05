using CSV
using DataFrames
using Plots

# Theoretical critical alpha for LSE (from phase_boundary_cpu.jl)
function critical_alpha_lse(T::Real)
    phi = 0.5 * (-T + sqrt(T^2 + 4))
    f_ret = (1 - phi) - (T/2) * log(1 - phi^2)
    alpha_c = 0.5 * (1 - f_ret)^2
    return clamp(alpha_c, 0.0, 0.5)
end

# Read the CSV file
data = CSV.read("lse_alpha_sweep_table.csv", DataFrame)

# Extract alpha values (first column)
alpha_values = data[:, 1]

# Extract T values from column names (excluding first column)
T_columns = names(data)[2:end]
T_values = [parse(Float64, replace(col, "T" => "")) for col in T_columns]

# Extract LSE values matrix (all columns except first)
lse_matrix = Matrix(data[:, 2:end])

# Create theoretical curve
T_theory = range(0.001, maximum(T_values), length=500)
alpha_c_theory = [critical_alpha_lse(T) for T in T_theory]

# Create heatmap with swapped axes (alpha on X-axis, T on Y-axis)
heatmap(alpha_values, T_values, lse_matrix',
        xlabel="α = ln(P)/N",
        ylabel="T",
        title="LSE (Gaussian Kernel)",
        colorbar_title="φ",
        color=cgrad([:darkblue, :blue, :white, :red, :darkred], rev=true),
        size=(800, 600))

# Add theoretical black line
plot!(alpha_c_theory, T_theory,
      color=:black,
      linewidth=2.5,
      label="Theoretical boundary",
      legend=:topright)

# Save the plot
savefig("lse_map.png")
println("Plot saved as lse_map.png")

# Display the plot (optional - comment out if running in non-interactive mode)
display(current())

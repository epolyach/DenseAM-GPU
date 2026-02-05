using CSV
using DataFrames
using Plots

# Theoretical critical alpha for LSR (from phase_boundary_cpu.jl)
function critical_alpha_lsr(T::Real, b::Real)
    A = b*T + 1
    B = -(2 + T + T*b)
    C = T
    disc = B^2 - 4*A*C

    disc < 0 && return NaN

    y = (-B - sqrt(disc)) / (2*A)
    phi = 1 - y

    (phi <= 1 - 1/b || phi > 1 || phi < 0) && return NaN

    u = -(1/b) * log(1 - b*(1-phi))
    s = 0.5 * log(1 - phi^2)
    f_ret = u - T * s

    alpha_c = 0.5 * (1 - f_ret)^2
    return clamp(alpha_c, 0.0, 0.5)
end

function find_T_max_lsr(b::Real, alpha_th::Real)
    for T in range(0.01, 3.0, length=1000)
        ac = critical_alpha_lsr(T, b)
        !isnan(ac) && ac <= alpha_th && return T
    end
    return NaN
end

# Parameters
b = 2 + sqrt(2)  # Epanechnikov parameter

# Read the CSV file
data = CSV.read("lsr_alpha_sweep_table.csv", DataFrame)

# Extract alpha values (first column)
alpha_values = data[:, 1]

# Extract T values from column names (excluding first column)
T_columns = names(data)[2:end]
T_values = [parse(Float64, replace(col, "T" => "")) for col in T_columns]

# Extract LSR values matrix (all columns except first)
lsr_matrix = Matrix(data[:, 2:end])

# Compute theoretical curves
T_theory = range(0.001, maximum(T_values), length=500)
alpha_c_lsr = [critical_alpha_lsr(T, b) for T in T_theory]

# Compute alpha_th and T_max for LSR
alpha_th = 0.5 * (1 - 1/b)^2
T_max = find_T_max_lsr(b, alpha_th)

# Create heatmap with swapped axes (alpha on X-axis, T on Y-axis)
heatmap(alpha_values, T_values, lsr_matrix',
        xlabel="α = ln(P)/N",
        ylabel="T",
        title="LSR (Epanechnikov, b=$(round(b, digits=2)))",
        colorbar_title="φ",
        color=cgrad([:darkblue, :blue, :white, :red, :darkred], rev=true),
        size=(800, 600))

# Add theoretical curves (with NaN filtering and constraints)
valid_idx = .!isnan.(alpha_c_lsr) .& (alpha_c_lsr .> alpha_th) .& (alpha_c_lsr .<= 0.5)
plot!(alpha_c_lsr[valid_idx], collect(T_theory)[valid_idx],
      color=:black,
      linewidth=2.5,
      label="Theoretical boundary")

# Add vertical line at alpha_th
if !isnan(T_max)
    plot!([alpha_th, alpha_th], [0, T_max],
          color=:black,
          linewidth=2.5,
          label=false)

    # Add horizontal line at T_max (dotted)
    plot!([alpha_th, maximum(alpha_values)], [T_max, T_max],
          color=:black,
          linewidth=1.5,
          linestyle=:dot,
          label=false)
end

# Add vertical line at α_c(0) = 0.5
vline!([0.5],
       color=:black,
       linewidth=1.5,
       linestyle=:dash,
       label=false,
       legend=:topright)

# Save the plot
savefig("lsr_map.png")
println("Plot saved as lsr_map.png")
println("α_th = $(round(alpha_th, digits=4))")
println("T_max = $(round(T_max, digits=4))")

# Display the plot (optional - comment out if running in non-interactive mode)
display(current())

using CSV
using DataFrames
using CairoMakie
using Printf

# ──────────────── Theoretical boundaries ────────────────

function critical_alpha_lse(T::Real)
    phi = 0.5 * (-T + sqrt(T^2 + 4))
    f_ret = (1 - phi) - (T/2) * log(1 - phi^2)
    alpha_c = 0.5 * (1 - f_ret)^2
    return clamp(alpha_c, 0.0, 0.5)
end

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

# ──────────────── Read data (Basin Stability) ────────────────

data_lse = CSV.read("basin_stab_LSE_v3.csv", DataFrame)
data_lsr = CSV.read("basin_stab_LSR_v3.csv", DataFrame)

# Extract α and T ranges separately for each dataset
alpha_lse = data_lse[:, 1]
T_cols_lse = names(data_lse)[2:end]
T_lse = [parse(Float64, replace(col, "T" => "")) for col in T_cols_lse]
lse_matrix = Matrix(data_lse[:, 2:end])

alpha_lsr = data_lsr[:, 1]
T_cols_lsr = names(data_lsr)[2:end]
T_lsr = [parse(Float64, replace(col, "T" => "")) for col in T_cols_lsr]
lsr_matrix = Matrix(data_lsr[:, 2:end])

# For shared colorscale and theory curves, use union of ranges
T_max_plot = max(maximum(T_lse), maximum(T_lsr))

# ──────────────── Theoretical curves ────────────────

b = 2 + sqrt(2)
alpha_th = 0.5 * (1 - 1/b)^2
T_max = find_T_max_lsr(b, alpha_th)

T_theory = range(0.001, T_max_plot, length=500)
alpha_c_lse = [critical_alpha_lse(T) for T in T_theory]
alpha_c_lsr = [critical_alpha_lsr(T, b) for T in T_theory]

# ──────────────── Production settings ────────────────

cmap = :RdBu
clims_val = (0.0, 1.0)

# Font sizes (sized for 0.9\textwidth figure)
fontsize_label = 26
fontsize_tick = 22
fontsize_colorbar = 22

# Tick values
alpha_major = collect(0.1:0.1:0.50)
alpha_minor = collect(0.05:0.05:0.55)
t_major = collect(0.0:0.5:2.0)
t_minor = collect(0.0:0.1:2.0)

# ──────────────── Create figure ────────────────

fig = Figure(size=(1600, 650), fontsize=fontsize_tick, figure_padding=10)

# Left panel: LSE
ax1 = Axis(fig[1, 1],
           xlabel="α = ln(P)/N", ylabel="T",
           xticks=alpha_major, yticks=t_major,
           xminorticks=alpha_minor, yminorticks=t_minor,
           xminorticksvisible=true, yminorticksvisible=true,
           xticklabelsize=fontsize_tick, yticklabelsize=fontsize_tick,
           xlabelsize=fontsize_label, ylabelsize=fontsize_label)

hm1 = heatmap!(ax1, alpha_lse, T_lse, lse_matrix,
               colormap=cmap, colorrange=clims_val)

# Theory curve
# lines!(ax1, alpha_c_lse, collect(T_theory), color=:black, linewidth=2.5)

# Dashed line at α=0.5
vlines!(ax1, [0.5], color=:black, linewidth=1.5, linestyle=:dash)

# Right panel: LSR
ax2 = Axis(fig[1, 2],
           xlabel="α = ln(P)/N", ylabel="",
           xticks=alpha_major, yticks=t_major,
           xminorticks=alpha_minor, yminorticks=t_minor,
           xminorticksvisible=true, yminorticksvisible=true,
           xticklabelsize=fontsize_tick, yticklabelsize=fontsize_tick,
           xlabelsize=fontsize_label, ylabelsize=fontsize_label,
           yticklabelsvisible=false)

hm2 = heatmap!(ax2, alpha_lsr, T_lsr, lsr_matrix,
               colormap=cmap, colorrange=clims_val)

"""
# Theory curve (curved portion)
valid_idx = .!isnan.(alpha_c_lsr) .& (alpha_c_lsr .> alpha_th) .& (alpha_c_lsr .<= 0.5)
lines!(ax2, alpha_c_lsr[valid_idx], collect(T_theory)[valid_idx],
       color=:black, linewidth=2.5)

# Theory vertical portion
if !isnan(T_max)
    lines!(ax2, [alpha_th, alpha_th], [T_max, maximum(T_lsr)],
           color=:black, linewidth=2.5)
end
"""

# Dashed line at α=0.5
vlines!(ax2, [0.5], color=:black, linewidth=1.5, linestyle=:dash)

# ──────────────── Colorbar ────────────────

cb = Colorbar(fig[1, 3], hm1,
              label="Alignment φ",
              labelsize=fontsize_label,
              ticklabelsize=fontsize_colorbar,
              ticks=([0.0, 0.25, 0.5, 0.75, 1.0], ["0.0", "0.25", "0.5", "0.75", "1.0"]),
              ticksvisible=true,
              width=45,
              flipaxis=true)   # ticks+labels on the right, label on the left

# ──────────────── Layout spacing ────────────────

colgap!(fig.layout, 1, 30)  # gap between LSE and LSR (pixels)
colgap!(fig.layout, 2, 30)  # gap between LSR and colorbar

# ──────────────── Save ────────────────

save("maps_basin_stab_prod.png", fig, px_per_unit=2)
println("✓ PNG saved: maps_basin_stab_prod.png")

save("maps_basin_stab_prod.eps", fig)
println("✓ EPS saved: maps_basin_stab_prod.eps")

println("\nBasin stability phase diagrams generated:")
println("  - maps_basin_stab_prod.png (raster, presentations)")
println("  - maps_basin_stab_prod.eps (vector, publication quality)")
println("\nProtocol: Retrieval test")
println("  Initial noise: ε = 0.25 (large perturbation)")
println("  Relaxation: 16384 steps (2^14)")
println("  Trials: 1024 (high statistics)")
println("\nInterpretation:")
println("  φ ≈ 1.0 (blue): Basin stable → successful retrieval")
println("  φ ≈ 0.0 (red):  Basin unstable → retrieval fails")
println("  LSR hard wall: φ_c ≈ 0.707 (prevents φ from reaching 0)")

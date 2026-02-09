using CSV
using DataFrames
using CairoMakie
using Printf

# ──────────────── Read pre-computed Boltzmann comparison data ────────────────

df = CSV.read("boltzmann_comparison_alpha01.csv", DataFrame)

# LSR parameters
b = 2 + sqrt(2)
phi_c = (b - 1) / b  # ≈ 0.707

# ──────────────── Production settings (1-column figure) ────────────────

fontsize_label = 26
fontsize_tick = 22
fontsize_legend = 20

fig = Figure(size=(800, 550), fontsize=fontsize_tick, figure_padding=15)

ax = Axis(fig[1, 1],
          xlabel="Temperature T", ylabel="Alignment φ",
          xlabelsize=fontsize_label, ylabelsize=fontsize_label,
          xticklabelsize=fontsize_tick, yticklabelsize=fontsize_tick,
          xticks=0.0:0.5:2.0,
          xminorticks=0.0:0.1:2.0, xminorticksvisible=true,
          yticks=0.7:0.05:1.0,
          yminorticks=0.7:0.01:1.0, yminorticksvisible=true,
          limits=(nothing, (0.7, 1.02)))

# Boltzmann theory curve (solid black)
lines!(ax, df.T, df.phi_theory,
       color=:black, linewidth=2.5,
       label="Boltzmann ⟨φ⟩")

# MC data points (blue circles)
scatter!(ax, df.T, df.phi_observed,
         color=:blue, markersize=8, marker=:circle,
         label="MC (α = 0.1)")

# Hard wall (dotted gray)
hlines!(ax, [phi_c],
        color=:gray40, linewidth=1.5, linestyle=:dot,
        label="φ_c = $(round(phi_c, digits=2))")

# Legend
axislegend(ax, position=:rt, labelsize=fontsize_legend, framevisible=true)

# ──────────────── Save ────────────────

save("boltzmann_comparison_alpha01.png", fig, px_per_unit=2)
println("✓ PNG saved: boltzmann_comparison_alpha01.png")

save("boltzmann_comparison_alpha01.eps", fig)
println("✓ EPS saved: boltzmann_comparison_alpha01.eps")

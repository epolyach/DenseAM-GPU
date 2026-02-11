using CSV
using DataFrames
using CairoMakie
using Printf

# ──────────────── Read MC data directly from basin stability CSV ────────────────

data = CSV.read("basin_stab_LSE_v3.csv", DataFrame)

T_cols = names(data)[2:end]
T_vals = [parse(Float64, replace(col, "T" => "")) for col in T_cols]

# Extract MC data for two α values
alpha_vals = [0.05, 0.10]
phi_mc = Dict{Float64, Vector{Float64}}()
for α in alpha_vals
    row = data[findfirst(data.alpha .== α), :]
    phi_mc[α] = Vector(row[2:end])
end

# ──────────────── Boltzmann integral (finite-N, single basin) ────────────────

# v3 power-law pattern count
const MIN_PAT = 20000
const MAX_PAT = 500000
const n_alpha_grid = 55
const pw_ind = 10
const alpha_grid = collect(0.01:0.01:0.55)
const n_patterns_vec = range(MIN_PAT^(1/pw_ind), MAX_PAT^(1/pw_ind), length=n_alpha_grid) .^ pw_ind

# Use the smallest α among MC cases → largest N (best thermodynamic limit)
alpha_boltz = minimum(alpha_vals)
boltz_idx = findfirst(alpha_grid .≈ alpha_boltz)
P_boltz = round(Int, n_patterns_vec[boltz_idx])
N_boltz = max(round(Int, log(P_boltz) / alpha_boltz), 2)

# LSE energy density: u(φ) = 1 - φ
# log ρ(φ) = ((N-2)/2) ln(1 - φ²)
# log P(φ) = log ρ + (-N(1-φ)/T)
function boltzmann_avg(N, T)
    n_pts = 5000
    phi_grid = range(-1.0 + 1e-10, 1.0 - 1e-10, length=n_pts)
    dphi = phi_grid[2] - phi_grid[1]

    log_probs = [((N-2)/2) * log(1 - φ^2) + N * φ / T for φ in phi_grid]
    lp_max = maximum(log_probs)

    w = @. exp(log_probs - lp_max)
    Z = sum(w) * dphi
    return sum(collect(phi_grid) .* w) * dphi / Z
end

T_fine = range(0.025, 2.0, length=200)
phi_theory = [boltzmann_avg(N_boltz, T) for T in T_fine]

@Printf.printf("Boltzmann curve: N = %d (α = %.2f, P = %d)\n", N_boltz, alpha_boltz, P_boltz)

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
          yticks=0.0:0.2:1.0,
          yminorticks=0.0:0.05:1.0, yminorticksvisible=true,
          limits=(nothing, (-0.05, 1.05)))

# Boltzmann theory curve (solid black)
lines!(ax, T_fine, phi_theory,
       color=:black, linewidth=2.5,
       label=rich("Boltzmann φ", subscript("eq")))

#        label=rich("Boltzmann φ", subscript("eq"), " (N=$N_boltz)"))

# MC data points
mc_colors = [:blue, :red]
mc_markers = [:circle, :utriangle]
for (i, α) in enumerate(alpha_vals)
    scatter!(ax, T_vals, phi_mc[α],
             color=mc_colors[i], markersize=8, marker=mc_markers[i],
             label="MC (α = $α)")
end

# Legend
axislegend(ax, position=:rt, labelsize=fontsize_legend, framevisible=true)

# ──────────────── Save ────────────────

save("boltzmann_comparison_LSE.png", fig, px_per_unit=2)
println("✓ PNG saved: boltzmann_comparison_LSE.png")

save("boltzmann_comparison_LSE.eps", fig)
println("✓ EPS saved: boltzmann_comparison_LSE.eps")

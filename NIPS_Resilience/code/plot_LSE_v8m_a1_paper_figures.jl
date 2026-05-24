#=
Publication figures from LSE v8m_a1 data
────────────────────────────────────────────────────────────────────────
Mirrors plot_v8m_paper_figures.jl but for the LSE energy with α extended
to [0.01, 1.00] (Ramsauer-boundary test for AAAI paper).

Theory:
  φ_eq(T)    = (1/2)(-T + √(T²+4))
  f_ret(T)   = 1 - φ_eq - (T/2) ln(1 - φ_eq²)
  α_c^G(T)   = (1/2)(1 - f_ret)²                    Gaussian
  α_c^E(T)   = -(1/2) ln(1 - (1-f_ret)²)            Exact density
  At T=0:    α_c^G = 0.5,    α_c^E = ∞

Input : basin_stab_LSE_v8m_a1.csv
Output: panels_paper/heatmap_LSE.{png,pdf}
        panels_paper/cdf_LSE_panel_A.{png,pdf}  (fixed T, varying α)
        panels_paper/cdf_LSE_panel_B.{png,pdf}  (fixed α, varying T)
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using Statistics

# ──────────────── Figure settings (match LSR paper) ────────────────
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)   # 86 mm
const FIG_H = round(Int, 56 / 25.4 * 100)   # 56 mm
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7
const FONT_LEG   = 6

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = "panels_paper"
mkpath(out_dir)

# ──────────────── Read data ────────────────
csv_in = "basin_stab_LSE_v8m_a1.csv"
println("Reading $csv_in...")
lines = readlines(csv_in)
n = length(lines) - 1
alpha = zeros(n); T = zeros(n); phi_a = zeros(n); phi_b = zeros(n); phimax = zeros(n)
for i in 1:n
    f = split(lines[i+1], ",")
    alpha[i] = parse(Float64, f[1]); T[i] = parse(Float64, f[2])
    phi_a[i] = parse(Float64, f[4]); phi_b[i] = parse(Float64, f[5])
    phimax[i] = parse(Float64, f[7])
end

alphas = sort(unique(round.(alpha, digits=4)))
Ts     = sort(unique(round.(T,     digits=4)))
na = length(alphas); nT = length(Ts)
@printf("Loaded: %d rows, %d α × %d T\n", n, na, nT)

# Disorder-averaged ⟨φ⟩ on the (α,T) grid (both replicas pooled)
phi_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 0.001) .& (abs.(T .- Ts[iT]) .< 0.001)
    any(mask) || continue
    phi_grid[iT, ia] = mean(vcat(phi_a[mask], phi_b[mask]))
end

# ──────────────── LSE theory ────────────────
φ_eq_LSE(T)  = 0.5 * (-T + sqrt(T^2 + 4))
s_func(φ)    = 0.5 * log(1 - φ^2)
f_ret_LSE(T) = let φ = φ_eq_LSE(T); 1 - φ - T*s_func(φ); end

α_c_gauss(T) = let fr = f_ret_LSE(T); fr >= 1 ? 0.0 : 0.5 * (1 - fr)^2; end
function α_c_exact(T)
    fr = f_ret_LSE(T)
    fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5 * log(arg)
end

# ──────────────── HEATMAP ────────────────
println("Plotting heatmap...")
p1 = heatmap(alphas, Ts, phi_grid,
    color=:RdBu, clims=(0, 1),
    xlabel="α", ylabel="T",
    xlims=(0, 1.0), ylims=(0, 2.0),
    colorbar_title="⟨φ⟩",
    size=(FIG_W + 40, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, bottom_margin=0Plots.mm)

# Theory overlays
T_th = range(0.005, 2.0, length=500)

# Gaussian: solid blue (finite at all T, → 0.5 as T → 0)
α_g = [α_c_gauss(t) for t in T_th]
plot!(p1, α_g, T_th, color=:blue, lw=2, label="Gaussian")

# Exact: dashed red (→ ∞ as T → 0; clip at right edge of plot)
α_e_raw = [α_c_exact(t) for t in T_th]
mask_e  = α_e_raw .<= 1.0
plot!(p1, α_e_raw[mask_e], T_th[mask_e], color=:red, lw=2, ls=:dash, label="Exact")

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "heatmap_LSE.$ext"))
end
println("Saved heatmap_LSE.{png,pdf}")

# ──────────────── CDF Panel A: fixed T, varying α ────────────────
# Use lowest T in grid (T = 0.025) to maximize Gaussian-vs-exact gap.
T_A = 0.025
cdf_points_A = [
    (0.10, T_A, "α=0.10"),
    (0.30, T_A, "α=0.30"),
    (0.50, T_A, "α=0.50"),  # Gaussian boundary at T=0
    (0.70, T_A, "α=0.70"),
    (0.90, T_A, "α=0.90"),
]

colors_cdf = [:darkblue, :dodgerblue, :green, :orange, :red, :darkred, :brown]

println("Plotting CDF panel A (fixed T=$T_A, varying α)...")
p2 = plot(xlabel="φ", ylabel="CDF",
    xlims=(-0.1, 1.05), ylims=(0, 1),
    legend=:topleft, legendfontsize=FONT_LEG,
    background_color_legend=RGBA(0.95,0.95,0.95,0.8),
    foreground_color_legend=RGBA(0.7,0.7,0.7,0.5),
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, bottom_margin=0Plots.mm,
    title=@sprintf("T = %.3f", T_A), titlefontsize=FONT_GUIDE)

# Reference: equilibrium φ_eq(T_A)
vline!(p2, [φ_eq_LSE(T_A)], color=:gray30, lw=1.5, ls=:dot, label="φ_eq(T)")

for (ci, (α_sel, T_sel, lbl)) in enumerate(cdf_points_A)
    mask = (abs.(alpha .- α_sel) .< 0.003) .& (abs.(T .- T_sel) .< 0.003)
    vals = sort(vcat(phi_a[mask], phi_b[mask]))
    isempty(vals) && (println("  no data at α=$α_sel T=$T_sel"); continue)
    cdf_y = range(0, 1, length=length(vals))
    plot!(p2, vals, cdf_y, color=colors_cdf[ci], lw=1.5, label=lbl)
end

for ext in ("png", "pdf")
    savefig(p2, joinpath(out_dir, "cdf_LSE_panel_A.$ext"))
end
println("Saved cdf_LSE_panel_A.{png,pdf}")

# ──────────────── CDF Panel B: fixed α, varying T ────────────────
# Pick α = 0.5: exactly on Gaussian boundary at T=0.
α_B = 0.50
cdf_points_B = [
    (α_B, 0.025, "T=0.025"),
    (α_B, 0.125, "T=0.125"),
    (α_B, 0.325, "T=0.325"),
    (α_B, 0.525, "T=0.525"),
    (α_B, 0.825, "T=0.825"),
    (α_B, 1.225, "T=1.225"),
]

colors_T = [:darkblue, :blue, :green, :orange, :red, :darkred]

println("Plotting CDF panel B (fixed α=$α_B, varying T)...")
p3 = plot(xlabel="φ", ylabel="CDF",
    xlims=(-0.1, 1.05), ylims=(0, 1),
    legend=:topleft, legendfontsize=FONT_LEG,
    background_color_legend=RGBA(0.95,0.95,0.95,0.8),
    foreground_color_legend=RGBA(0.7,0.7,0.7,0.5),
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, bottom_margin=0Plots.mm,
    title=@sprintf("α = %.2f", α_B), titlefontsize=FONT_GUIDE)

# Show φ_eq(T) for each selected T as a faint dotted vline
for (_, T_sel, _) in cdf_points_B
    vline!(p3, [φ_eq_LSE(T_sel)], color=:gray70, lw=0.8, ls=:dot, label=false)
end

for (ci, (α_sel, T_sel, lbl)) in enumerate(cdf_points_B)
    mask = (abs.(alpha .- α_sel) .< 0.003) .& (abs.(T .- T_sel) .< 0.003)
    vals = sort(vcat(phi_a[mask], phi_b[mask]))
    isempty(vals) && (println("  no data at α=$α_sel T=$T_sel"); continue)
    cdf_y = range(0, 1, length=length(vals))
    plot!(p3, vals, cdf_y, color=colors_T[ci], lw=1.5, label=lbl)
end

for ext in ("png", "pdf")
    savefig(p3, joinpath(out_dir, "cdf_LSE_panel_B.$ext"))
end
println("Saved cdf_LSE_panel_B.{png,pdf}")

# ──────────────── Statistics ────────────────
println("\nDetailed statistics at T = $T_A (lowest T):")
@printf("  %-5s  %5s  %6s  %6s  %6s  %6s\n",
    "α", "n_dis", "⟨φ⟩", "σ(φ)", "frac<0.5", "frac<0.9·φ_eq")
println("  " * "─"^55)
φ_eq_A = φ_eq_LSE(T_A)
for α_sel in [0.10, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    mask = (abs.(alpha .- α_sel) .< 0.003) .& (abs.(T .- T_A) .< 0.003)
    pa = phi_a[mask]; pb = phi_b[mask]
    vals = vcat(pa, pb)
    isempty(vals) && continue
    ndis = length(pa)
    @printf("  %.2f   %4d  %6.3f  %5.3f    %5.3f    %5.3f\n",
        α_sel, ndis, mean(vals), std(vals),
        mean(vals .< 0.5), mean(vals .< 0.9*φ_eq_A))
end

println("\nGaussian boundary α_c^G(0) = 0.500")
println("Exact   boundary α_c^E(0) = ∞")
println("\nDone.")

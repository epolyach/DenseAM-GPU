#=
Plot LSE phase boundary: Gaussian vs Exact alignment distribution
────────────────────────────────────────────────────────────────────────
Output: panels_LSE/phase_boundary_comparison.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf

# ──────────────── Theory ────────────────

# Retrieval equilibrium overlap (ICML Eq. 33) — exact, no approximation
φ_LSE(T) = 0.5 * (-T + sqrt(T^2 + 4))

# Retrieval free energy density (ICML Eq. 34) — exact
f_ret(T) = let φ = φ_LSE(T); 1 - φ - (T/2)*log(1 - φ^2); end

# Phase boundary: Gaussian (ICML Eq. 35)
α_c_gauss(T) = 0.5 * (1 - f_ret(T))^2

# Phase boundary: Exact density
function α_c_exact(T)
    fr = f_ret(T)
    arg = fr * (2 - fr)  # = 1 - (1-fr)²
    arg ≤ 0 && return Inf
    return -0.5 * log(arg)
end

# ──────────────── Plot ────────────────

# 86mm figure, same style as centroid_geometry etc.
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)   # 339 px
const FIG_H = FIG_W
const FONT_TITLE = 9
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = "panels_LSE"
mkpath(out_dir)

T_range = range(0.005, 2.0, length=500)
αg = [α_c_gauss(T) for T in T_range]
αe = [min(α_c_exact(T), 1.05) for T in T_range]  # cap just above xlim

# Gaussian curve: clip to α ≤ 0.5 (it ends there by definition)
T_gauss = [T for T in T_range if α_c_gauss(T) ≤ 0.5]
α_gauss = [α_c_gauss(T) for T in T_gauss]

p = plot(α_gauss, T_gauss,
    color=:blue, lw=2, label="Gaussian",
    xlabel="α", ylabel="T",
    xlims=(0, 1.0), ylims=(0, 2.0),
    legend=:topright, legendfontsize=FONT_ANN,
    background_color_legend=RGBA(0.95,0.95,0.95,0.8),
    foreground_color_legend=RGBA(0.7,0.7,0.7,0.5),
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, right_margin=0Plots.mm,
    top_margin=0Plots.mm, bottom_margin=0Plots.mm)

plot!(p, αe, T_range,
    color=:red, lw=2, ls=:dash, label="Exact")

# Region labels — Retrieval below curves
annotate!(p, 0.12, 0.20, text("Retrieval", FONT_ANN+1, :center, :black))
annotate!(p, 0.70, 1.0, text("Non-retrieval", FONT_ANN+1, :center, :black))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "phase_boundary_comparison.$ext"))
end
println("Saved: $(out_dir)/phase_boundary_comparison.{png,pdf}")

# ──────────────── Print table ────────────────
println()
println("  T       φ_eq     f_ret    α_c^Gauss   α_c^exact   ratio")
println("  " * "─"^58)
for T in [0.001, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
    fr = f_ret(T)
    ag = α_c_gauss(T)
    ae = α_c_exact(T)
    @printf("  %.3f   %.4f   %.4f    %.4f      %.4f     %.2f\n",
        T, φ_LSE(T), fr, ag, ae > 10 ? Inf : ae, ae/ag)
end

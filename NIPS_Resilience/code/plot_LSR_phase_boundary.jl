#=
Plot LSR phase boundary: Gaussian vs Exact alignment distribution
────────────────────────────────────────────────────────────────────────
Output: panels_LSR/phase_boundary_comparison.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf

# ──────────────── Parameters ────────────────

const b = 2 + sqrt(2)
const φ_c = (b - 1) / b   # ≈ 0.7071

# ──────────────── LSR retrieval theory ────────────────

# LSR equilibrium overlap: solve (bT+1)y² - (2+T+Tb)y + T = 0 where y = 1-φ
# From ICML Eq. 37
function φ_LSR(T)
    a_coeff = b*T + 1
    b_coeff = -(2 + T + T*b)
    c_coeff = T
    disc = b_coeff^2 - 4*a_coeff*c_coeff
    disc < 0 && return NaN
    y = (-b_coeff - sqrt(disc)) / (2*a_coeff)  # smaller root → larger φ
    return 1 - y
end

# LSR internal energy density (ICML Eq. 14)
u_LSR(φ) = -log(1 - b*(1-φ)) / b

# Entropy (same for both kernels)
s(φ) = 0.5 * log(1 - φ^2)

# Retrieval free energy
f_ret_LSR(T) = u_LSR(φ_LSR(T)) - T * s(φ_LSR(T))

# ──────────────── Phase boundaries ────────────────

# Gaussian: α_c = ½(1-f_ret)², with vertical asymptote at α_th = φ_c²/2
α_th_gauss = φ_c^2 / 2
α_c_gauss(T) = 0.5 * (1 - f_ret_LSR(T))^2

# Exact: α_c = -½ln(1-(1-f_ret)²), with vertical asymptote at α_th = -½ln(1-φ_c²)
α_th_exact = -0.5 * log(1 - φ_c^2)
function α_c_exact(T)
    fr = f_ret_LSR(T)
    arg = fr * (2 - fr)
    arg ≤ 0 && return Inf
    return -0.5 * log(arg)
end

# ──────────────── Figure ────────────────

const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = FIG_W
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = "panels_LSR"
mkpath(out_dir)

T_range = range(0.005, 2.0, length=500)

# Only the physical branch: f_ret ≤ 1 (above this, curve goes back up — artifact)
T_phys = [T for T in T_range if f_ret_LSR(T) ≤ 1.0]

# Split into solid (α > α_th) and dotted (α < α_th)
T_solid_g = [T for T in T_phys if α_c_gauss(T) ≥ α_th_gauss && α_c_gauss(T) ≤ 1.0]
α_solid_g = [α_c_gauss(T) for T in T_solid_g]
T_dot_g   = [T for T in T_phys if α_c_gauss(T) < α_th_gauss]
α_dot_g   = [α_c_gauss(T) for T in T_dot_g]

T_solid_e = [T for T in T_phys if α_c_exact(T) ≥ α_th_exact && α_c_exact(T) ≤ 1.0]
α_solid_e = [α_c_exact(T) for T in T_solid_e]
T_dot_e   = [T for T in T_phys if α_c_exact(T) < α_th_exact]
α_dot_e   = [min(α_c_exact(T), 1.0) for T in T_dot_e]

p = plot(xlabel="α", ylabel="T",
    xlims=(0, 1.0), ylims=(0, 2.0),
    legend=:topright, legendfontsize=FONT_ANN,
    background_color_legend=RGBA(0.95,0.95,0.95,0.8),
    foreground_color_legend=RGBA(0.7,0.7,0.7,0.5),
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, right_margin=0Plots.mm,
    top_margin=0Plots.mm, bottom_margin=0Plots.mm)

# Find T where curves meet the vertical lines
T_join_g = isempty(T_solid_g) ? 0.0 : maximum(T_solid_g)
T_join_e = isempty(T_solid_e) ? 0.0 : maximum(T_solid_e)

# Gaussian: vertical line from T_join to top, solid curve (α > α_th), dotted (α < α_th)
plot!(p, [α_th_gauss, α_th_gauss], [T_join_g, 2.0], color=:blue, lw=2, label=false)
plot!(p, α_solid_g, T_solid_g, color=:blue, lw=2, label="Gaussian")
plot!(p, α_dot_g, T_dot_g, color=:blue, lw=1.5, ls=:dot, label=false)

# Exact: vertical line from T_join to top, solid curve (α > α_th), dotted (α < α_th)
plot!(p, [α_th_exact, α_th_exact], [T_join_e, 2.0], color=:red, lw=2, ls=:dash, label=false)
plot!(p, α_solid_e, T_solid_e, color=:red, lw=2, ls=:dash, label="Exact")
plot!(p, α_dot_e, T_dot_e, color=:red, lw=1.5, ls=:dot, label=false)

# Region labels
annotate!(p, 0.12, 0.20, text("Retrieval", FONT_ANN+1, :center, :black))
annotate!(p, 0.70, 1.0, text("Non-retrieval", FONT_ANN+1, :center, :black))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "phase_boundary_comparison.$ext"))
end
println("Saved: $(out_dir)/phase_boundary_comparison.{png,pdf}")

# ──────────────── Print key values ────────────────
println()
@printf("  α_th^Gauss = %.4f,  α_th^exact = %.4f  (ratio %.2f)\n",
    α_th_gauss, α_th_exact, α_th_exact/α_th_gauss)
println()
println("  T       φ_eq     f_ret    α_c^Gauss   α_c^exact")
println("  " * "─"^50)
for T in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]
    fr = f_ret_LSR(T)
    ag = α_c_gauss(T)
    ae = α_c_exact(T)
    @printf("  %.2f    %.4f   %.4f    %.4f      %.4f\n", T, φ_LSR(T), fr, ag, min(ae, 99.0))
end

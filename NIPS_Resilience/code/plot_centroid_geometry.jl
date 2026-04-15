#=
Geometric illustration of the centroid escape mechanism
────────────────────────────────────────────────────────────────────────
Shows the 2D projection of the N-sphere in the (ξ¹, ξ^μ) plane:
  - Unit disk = feasible region (a² + b² ≤ 1)
  - Shaded: support regions for ξ¹ and ξ^μ
  - Overlap region where both patterns are in support
  - Retrieval state, centroid, and zero-barrier escape path

The coordinates (a, b) are:
  a = φ₁ = (ξ¹·x)/N   (overlap with target)
  b = component along ξ^μ_⊥ (orthogonalized)
  φ_μ = a·q + b·√(1-q²)

Input:  none (standalone)
Output: panels_v8m/centroid_geometry.png, .pdf
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf

# ──────────────── Parameters ────────────────

const b_lsr = 2 + sqrt(2)
const phi_c = (b_lsr - 1) / b_lsr   # ≈ 0.707
const q = 0.38                        # inter-pattern overlap
const sq = sqrt(1 - q^2)             # ≈ 0.925

# Key points
const phi_eq = 0.90                   # retrieval equilibrium
const phi_cen = 0.831                 # centroid overlap with each pattern

# Retrieval state: (a, b) = (φ_eq, ~0) — perpendicular component is random, ~0 in this plane
const a_ret = phi_eq
const b_ret = 0.0

# Centroid: equal overlap with both patterns → φ₁ = φ_μ = phi_cen
# φ_μ = a·q + b·sq = phi_cen, φ₁ = a = phi_cen
const a_cen = phi_cen
const b_cen = (phi_cen - a_cen * q) / sq   # = phi_cen(1-q)/sq

# Entry point: where ξ^μ enters support along the escape path (a = φ_eq fixed)
# φ_μ = a_ret·q + b_entry·sq = phi_c
const b_entry = (phi_c - a_ret * q) / sq

# Figure size & fonts (identical to plot_retrieval_prob_v8m.jl / plot_panels_LSR_v8.jl)
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)   # 339 logical pixels (86mm at 100 base DPI)
const FIG_H = round(Int, FIG_W * 0.85)       # 288

const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 7
const FONT_ANNOT  = 6

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND,
        colorbar_tickfontsize=FONT_TICK, colorbar_titlefontsize=FONT_GUIDE)

out_dir = "panels_v8m"
mkpath(out_dir)

# ──────────────── Helper: fill region ────────────────

# Unit circle boundary
θ_circ = range(0, 2π, length=500)
circ_a = cos.(θ_circ)
circ_b = sin.(θ_circ)

# ──────────────── Plot ────────────────

p = plot(size=(FIG_W, FIG_H), dpi=FIG_DPI, aspect_ratio=:equal,
    xlabel="a = φ₁", ylabel="b",
    title="Support geometry in (ξ¹, ξ^μ) plane",
    xlims=(-0.15, 1.15), ylims=(-0.55, 0.75),
    legend=:bottomleft, legendfontsize=FONT_LEGEND,
    right_margin=5Plots.mm, left_margin=3Plots.mm,
    top_margin=3Plots.mm, bottom_margin=3Plots.mm)

# Unit circle (sphere constraint)
plot!(p, circ_a, circ_b, color=:black, lw=1.5, ls=:solid, label="a² + b² = 1")

# ── Support region for ξ¹: a > φ_c ──
# Fill the region inside the circle with a > φ_c
a_fill1 = range(phi_c, 1.0, length=200)
b_upper1 = sqrt.(max.(0, 1.0 .- a_fill1.^2))
b_lower1 = -b_upper1
plot!(p, vcat(a_fill1, reverse(a_fill1)),
    vcat(b_upper1, reverse(b_lower1)),
    fillrange=0, fillalpha=0.15, fillcolor=:blue, lw=0, label="")
vline!(p, [phi_c], color=:blue, lw=1.5, ls=:dash,
    label=@sprintf("ξ¹ support: φ₁ > φ_c = %.3f", phi_c))

# ── Support region for ξ^μ: a·q + b·sq > φ_c ──
# Line: b = (φ_c - a·q)/sq
a_line = range(-0.2, 1.15, length=200)
b_line = (phi_c .- a_line .* q) ./ sq

# Fill the region inside the circle above this line
# Clip to circle
a_fill2 = Float64[]
b_upper2 = Float64[]
b_lower2 = Float64[]
for a in range(-0.2, 1.0, length=300)
    b_supp = (phi_c - a * q) / sq
    b_circ_upper = sqrt(max(0, 1.0 - a^2))
    b_circ_lower = -b_circ_upper
    if b_supp < b_circ_upper && a^2 <= 1.0
        push!(a_fill2, a)
        push!(b_upper2, min(b_circ_upper, 0.75))
        push!(b_lower2, max(b_supp, b_circ_lower))
    end
end
plot!(p, vcat(a_fill2, reverse(a_fill2)),
    vcat(b_upper2, reverse(b_lower2)),
    fillrange=0, fillalpha=0.12, fillcolor=:red, lw=0, label="")
plot!(p, a_line, b_line, color=:red, lw=1.5, ls=:dash,
    xlims=(-0.15, 1.15), ylims=(-0.55, 0.75),
    label=@sprintf("ξ^μ support: φ_μ > φ_c  (q=%.2f)", q))

# ── Overlap region label ──
annotate!(p, 0.88, 0.32, text("overlap\nregion", FONT_ANNOT, :center, :gray30))

# ── Key points ──

# Retrieval state
scatter!(p, [a_ret], [b_ret], color=:green, markersize=8, markershape=:star5,
    markerstrokewidth=1, markerstrokecolor=:black,
    label=@sprintf("retrieval (%.2f, %.2f)", a_ret, b_ret))

# Centroid
scatter!(p, [a_cen], [b_cen], color=:orange, markersize=8, markershape=:diamond,
    markerstrokewidth=1, markerstrokecolor=:black,
    label=@sprintf("centroid (%.2f, %.2f)", a_cen, b_cen))

# Entry point (where ξ^μ enters support)
scatter!(p, [a_ret], [b_entry], color=:red, markersize=6, markershape=:circle,
    markerstrokewidth=1, markerstrokecolor=:black,
    label=@sprintf("ξ^μ enters support (%.2f, %.2f)", a_ret, b_entry))

# ── Escape path ──
# Step 1: vertical (rotate perpendicular, φ₁ fixed)
plot!(p, [a_ret, a_ret], [b_ret, b_entry],
    color=:black, lw=2, ls=:solid, arrow=true, label="")
annotate!(p, a_ret + 0.03, (b_ret + b_entry)/2,
    text("① rotate x_⊥\n(ΔE = 0)", FONT_ANNOT, :left, :black))

# Step 2: slide to centroid (both patterns in support, energy decreases)
plot!(p, [a_ret, a_cen], [b_entry, b_cen],
    color=:black, lw=2, ls=:solid, arrow=true, label="")
annotate!(p, (a_ret + a_cen)/2 - 0.07, (b_entry + b_cen)/2 + 0.03,
    text("② slide to centroid\n(ΔE < 0)", FONT_ANNOT, :left, :black))

# ── Energy annotations ──
annotate!(p, a_ret - 0.02, b_ret - 0.06,
    text(@sprintf("E/N = %.3f", -(1/b_lsr)*log(max(0, 1-b_lsr+b_lsr*phi_eq))),
    FONT_ANNOT, :right, :green4))

s_cen = 2 * max(0, 1 - b_lsr + b_lsr * phi_cen)
annotate!(p, a_cen + 0.02, b_cen + 0.04,
    text(@sprintf("E/N = %.3f", -(1/b_lsr)*log(s_cen)),
    FONT_ANNOT, :left, :darkorange))

# Save
for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "centroid_geometry.$ext"))
end
println("Saved: $(out_dir)/centroid_geometry.{png,pdf}")

# Print key numbers for the tex file
println()
println("=== Key numbers ===")
println("b = $(round(b_lsr, digits=4))")
println("phi_c = $(round(phi_c, digits=4))")
println("q = $q, sqrt(1-q²) = $(round(sq, digits=4))")
println("Retrieval: ($(a_ret), $(b_ret)), phi_mu = $(round(a_ret*q, digits=3))")
println("Entry point: ($(a_ret), $(round(b_entry, digits=3)))")
println("Centroid: ($(round(a_cen, digits=3)), $(round(b_cen, digits=3)))")
println("E_ret/N = $(round(-(1/b_lsr)*log(max(0, 1-b_lsr+b_lsr*phi_eq)), digits=4))")
println("E_cen/N = $(round(-(1/b_lsr)*log(s_cen), digits=4))")
println("phi_mu at entry = $(round(a_ret*q + b_entry*sq, digits=4))")

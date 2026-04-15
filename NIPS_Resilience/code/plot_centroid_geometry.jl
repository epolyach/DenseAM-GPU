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

const a_ret = phi_eq
const b_ret = 0.0

const a_cen = phi_cen
const b_cen = (phi_cen - a_cen * q) / sq

const b_entry = (phi_c - a_ret * q) / sq

# Figure: 86mm width, square aspect for geometry
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)   # 339
const FIG_H = FIG_W                           # square

const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 6
const FONT_ANNOT  = 7

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND,
        colorbar_tickfontsize=FONT_TICK, colorbar_titlefontsize=FONT_GUIDE)

out_dir = "panels_v8m"
mkpath(out_dir)

# ──────────────── Plot ────────────────

θ_circ = range(0, 2π, length=500)
circ_a = cos.(θ_circ)
circ_b = sin.(θ_circ)

p = plot(size=(FIG_W, FIG_H), dpi=FIG_DPI, aspect_ratio=:equal,
    xlabel="a = φ₁", ylabel="b",
    xlims=(0.55, 1.05), ylims=(-0.15, 0.65),
    legend=:topleft, legendfontsize=FONT_LEGEND,
    left_margin=1Plots.mm, right_margin=1Plots.mm,
    top_margin=1Plots.mm, bottom_margin=1Plots.mm)

# Unit circle
plot!(p, circ_a, circ_b, color=:black, lw=1.5, label="a²+b²=1")

# ── Support ξ¹: a > φ_c ──
a_fill1 = range(phi_c, 1.0, length=200)
b_up1 = sqrt.(max.(0, 1.0 .- a_fill1.^2))
b_lo1 = -b_up1
plot!(p, vcat(a_fill1, reverse(a_fill1)), vcat(b_up1, reverse(b_lo1)),
    fillrange=0, fillalpha=0.15, fillcolor=:blue, lw=0, label="")
vline!(p, [phi_c], color=:blue, lw=1.5, ls=:dash, label="ξ¹ support (φ₁>φ_c)")

# ── Support ξ^μ: a·q + b·sq > φ_c ──
a_line = range(0.5, 1.05, length=200)
b_line = (phi_c .- a_line .* q) ./ sq

a_f2 = Float64[]; b_u2 = Float64[]; b_l2 = Float64[]
for a in range(0.5, 1.0, length=300)
    bs = (phi_c - a * q) / sq
    bc = sqrt(max(0, 1.0 - a^2))
    if bs < bc && a^2 <= 1.0
        push!(a_f2, a); push!(b_u2, min(bc, 0.65)); push!(b_l2, max(bs, -bc))
    end
end
plot!(p, vcat(a_f2, reverse(a_f2)), vcat(b_u2, reverse(b_l2)),
    fillrange=0, fillalpha=0.12, fillcolor=:red, lw=0, label="")
plot!(p, a_line, b_line, color=:red, lw=1.5, ls=:dash, label="ξ^μ support (q=0.38)")

# ── Key points ──
scatter!(p, [a_ret], [b_ret], color=:green, markersize=7, markershape=:star5,
    markerstrokewidth=1, markerstrokecolor=:black, label="retrieval")
scatter!(p, [a_cen], [b_cen], color=:orange, markersize=7, markershape=:diamond,
    markerstrokewidth=1, markerstrokecolor=:black, label="centroid")
scatter!(p, [a_ret], [b_entry], color=:red, markersize=5, markershape=:circle,
    markerstrokewidth=1, markerstrokecolor=:black, label="entry point")

# ── Escape path arrows ──
plot!(p, [a_ret, a_ret], [b_ret, b_entry],
    color=:black, lw=2.5, arrow=true, label="")
plot!(p, [a_ret, a_cen], [b_entry, b_cen],
    color=:black, lw=2.5, arrow=true, label="")

# ── Annotations (positioned to avoid overlap) ──
annotate!(p, a_ret - 0.015, (b_ret + b_entry)/2,
    text("①  ΔE=0", FONT_ANNOT, :right, :black))
annotate!(p, (a_ret + a_cen)/2 + 0.01, (b_entry + b_cen)/2 + 0.04,
    text("②  ΔE<0", FONT_ANNOT, :left, :black))

annotate!(p, a_ret, b_ret - 0.04,
    text(@sprintf("E/N=%.3f", -(1/b_lsr)*log(max(0,1-b_lsr+b_lsr*phi_eq))),
    FONT_ANNOT-1, :center, :green4))

s_cen = 2 * max(0, 1 - b_lsr + b_lsr * phi_cen)
annotate!(p, a_cen - 0.01, b_cen + 0.04,
    text(@sprintf("E/N=%.3f", -(1/b_lsr)*log(s_cen)),
    FONT_ANNOT-1, :right, :darkorange))

# Save
for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "centroid_geometry.$ext"))
end
println("Saved: $(out_dir)/centroid_geometry.{png,pdf}")

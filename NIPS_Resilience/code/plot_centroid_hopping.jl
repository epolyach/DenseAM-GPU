#=
Centroid hopping schematic
────────────────────────────────────────────────────────────────────────
Shows the relay mechanism: the system hops between two-pattern centroids
along the pattern overlap graph, rather than forming higher-order centroids.

Output: panels_v8m/centroid_hopping.png, .pdf
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf

# ──────────────── Parameters ────────────────

const b_lsr = 2 + sqrt(2)
const phi_c = (b_lsr - 1) / b_lsr
const q = 0.38

# Centroid overlap: φ_cen = √((1+q)/2)
const phi_cen = sqrt((1 + q) / 2)

# Energy at 2-pattern centroid
const E_cen = -(1/b_lsr) * log(2 * max(0, 1 - b_lsr + b_lsr * phi_cen))
const E_ret = -(1/b_lsr) * log(max(0, 1 - b_lsr + b_lsr * 0.90))

# Target overlap at the spurious centroid (μ,ν):
# φ_1 = (q_{1μ} + q_{1ν}) / √(2(1+q_{μν}))
const phi_1_at_spurious = (q + q) / sqrt(2*(1+q))  # both q_{1μ}=q_{1ν}=q
# More realistic: q_{1ν} is random ~ 0, so
const phi_1_at_spurious_asym = (q + 0.10) / sqrt(2*(1+q))

# Figure settings
const FIG_DPI = 300
const FIG_W = round(Int, 170 / 25.4 * 100)  # full width
const FIG_H = round(Int, 70 / 25.4 * 100)

const FONT_GUIDE = 9
const FONT_TICK  = 7
const FONT_ANN   = 8
const FONT_SMALL = 7

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = "panels_v8m"
mkpath(out_dir)

# ──────────────── Layout ────────────────

# Pattern positions along x-axis
x_pat = [0.0, 2.0, 4.0, 6.0]   # ξ^1, ξ^μ, ξ^ν, ξ^ρ
y_pat = 0.0

# Centroid positions (midpoints of edges, slightly above)
x_cen = [1.0, 3.0, 5.0]         # (1,μ), (μ,ν), (ν,ρ)
y_cen = 0.0

# Retrieval position (at ξ^1)
x_ret = 0.0

# ──────────────── Plot ────────────────

p = plot(size=(FIG_W, FIG_H), dpi=FIG_DPI,
    xlims=(-0.8, 7.5), ylims=(-1.6, 2.2),
    legend=false, axis=false, grid=false, ticks=false,
    left_margin=2Plots.mm, right_margin=2Plots.mm,
    top_margin=0Plots.mm, bottom_margin=0Plots.mm)

# ── Pattern overlap edges ──
for i in 1:3
    plot!(p, [x_pat[i], x_pat[i+1]], [0, 0], color=:gray70, lw=3, label=false)
    # Edge label: q
    annotate!(p, (x_pat[i]+x_pat[i+1])/2, 0.35,
        text("q≈0.38", FONT_SMALL, :center, :gray50))
end
# Dots continuing
annotate!(p, 6.7, 0.0, text("···", 14, :center, :gray50))

# ── Pattern nodes ──
pat_labels = ["ξ¹", "ξᵘ", "ξᵛ", "ξᵖ"]
pat_colors = [:green4, :royalblue, :darkorange, :purple]
for i in 1:4
    scatter!(p, [x_pat[i]], [0], markersize=14, color=pat_colors[i],
        markerstrokewidth=1.5, markerstrokecolor=:black, label=false)
    annotate!(p, x_pat[i], 0.7, text(pat_labels[i], FONT_ANN+1, :center, pat_colors[i]))
end

# ── Retrieval state ──
scatter!(p, [x_ret], [-0.05], markersize=5, color=:green,
    markershape=:star5, markerstrokewidth=0.5, label=false)

# ── Centroids (diamonds on edges) ──
cen_labels = ["(ξ¹,ξᵘ)", "(ξᵘ,ξᵛ)", "(ξᵛ,ξᵖ)"]
cen_colors = [:teal, :chocolate, :mediumpurple]
for i in 1:3
    scatter!(p, [x_cen[i]], [0], markersize=10, markershape=:diamond,
        color=cen_colors[i], markerstrokewidth=1, markerstrokecolor=:black, label=false)
end

# ── Hopping arrows (curved, above) ──
# Arrow from retrieval to centroid (1,μ)
arrow_y = 1.3
for (i, (x1, x2)) in enumerate([(0.15, 0.85), (1.15, 2.85), (3.15, 4.85)])
    xm = (x1 + x2) / 2
    # Curved arrow via 3 points
    xx = range(x1, x2, length=30)
    yy = arrow_y .+ 0.25 .* sin.(π .* (xx .- x1) ./ (x2 - x1))
    plot!(p, xx, yy, color=:black, lw=2, arrow=(:closed, 2.0), label=false)

    # Hop labels
    hop_label = i == 1 ? "1st hop" : (i == 2 ? "2nd hop" : "3rd hop")
    annotate!(p, xm, arrow_y + 0.55, text(hop_label, FONT_SMALL, :center, :black))
end

# ── Info below: φ_1 at each position ──
info_y = -0.9

# At retrieval
annotate!(p, 0.0, info_y, text("φ₁=0.90", FONT_SMALL, :center, :green4))
annotate!(p, 0.0, info_y - 0.4, text("E/N=0.12", FONT_SMALL, :center, :green4))

# At centroid (1,μ)
annotate!(p, 1.0, info_y, text("φ₁=0.83", FONT_SMALL, :center, :teal))
annotate!(p, 1.0, info_y - 0.4, text("E/N=0.05", FONT_SMALL, :center, :teal))

# At centroid (μ,ν)
annotate!(p, 3.0, info_y,
    text(@sprintf("φ₁≈%.2f", phi_1_at_spurious), FONT_SMALL, :center, :chocolate))
annotate!(p, 3.0, info_y - 0.4, text("E/N=0.05", FONT_SMALL, :center, :chocolate))

# At centroid (ν,ρ)
annotate!(p, 5.0, info_y, text("φ₁→0", FONT_SMALL, :center, :mediumpurple))
annotate!(p, 5.0, info_y - 0.4, text("E/N=0.05", FONT_SMALL, :center, :mediumpurple))

# ── Bracket: "target in support" vs "target NOT in support" ──
plot!(p, [-0.3, 1.5], [-1.55, -1.55], color=:blue, lw=1.5, label=false)
annotate!(p, 0.6, -1.55, text("▲", 6, :center, :blue))
annotate!(p, 0.6, -1.85, text("target in support", FONT_SMALL-1, :center, :blue))

plot!(p, [2.0, 6.5], [-1.55, -1.55], color=:red, lw=1.5, label=false)
annotate!(p, 4.25, -1.55, text("▲", 6, :center, :red))
annotate!(p, 4.25, -1.85, text("target NOT in support", FONT_SMALL-1, :center, :red))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "centroid_hopping.$ext"))
end
println("Saved: $(out_dir)/centroid_hopping.{png,pdf}")

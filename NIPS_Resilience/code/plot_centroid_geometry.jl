#=
Geometric illustration of the centroid escape mechanism
────────────────────────────────────────────────────────────────────────
Output: panels_v8m/centroid_geometry.png, .pdf
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf

# ──────────────── Parameters ────────────────

const b_lsr = 2 + sqrt(2)
const phi_c = (b_lsr - 1) / b_lsr
const q = 0.38
const sq = sqrt(1 - q^2)

const phi_eq = 0.90
const phi_cen = 0.831

const a_ret = phi_eq;         const b_ret = 0.0
const a_cen = phi_cen;        const b_cen = (phi_cen - a_cen * q) / sq
const b_entry = (phi_c - a_ret * q) / sq

# Figure: 86mm width, square
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)   # 339
const FIG_H = FIG_W

const FONT_TITLE = 9
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = "panels_v8m"
mkpath(out_dir)

# ──────────────── Plot ────────────────

θ = range(0, 2π, length=500)

p = plot(cos.(θ), sin.(θ), color=:black, lw=1.5, label=false,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    xlabel="a = φ₁", ylabel="b",
    xlims=(0.50, 1.10), ylims=(-0.05, 0.55),
    legend=false,
    left_margin=0Plots.mm, right_margin=0Plots.mm,
    top_margin=0Plots.mm, bottom_margin=0Plots.mm)

# ── Filled support regions ──

# ξ¹ support: a > φ_c (blue)
af = range(phi_c, 1.0, length=200)
bu = sqrt.(max.(0, 1.0 .- af.^2))
plot!(p, vcat(af, reverse(af)), vcat(bu, -reverse(bu)),
    fillrange=0, fillalpha=0.15, fillcolor=:blue, lw=0, label=false)

# ξ^μ support: aq + b·sq > φ_c (red)
for a in range(0.55, 1.0, length=300)
    bs = (phi_c - a * q) / sq
    bc = sqrt(max(0, 1.0 - a^2))
    if bs < bc && a^2 <= 1.0
        # just build the fill inline via vertical segments
    end
end
# simpler: fill polygon for ξ^μ support clipped to circle
a_f = Float64[]; b_u = Float64[]; b_l = Float64[]
for a in range(0.50, 1.0, length=400)
    bs = (phi_c - a * q) / sq
    bc = sqrt(max(0, 1.0 - a^2))
    if bs < bc
        push!(a_f, a); push!(b_u, min(bc, 0.55)); push!(b_l, max(bs, -bc))
    end
end
plot!(p, vcat(a_f, reverse(a_f)), vcat(b_u, reverse(b_l)),
    fillrange=0, fillalpha=0.12, fillcolor=:red, lw=0, label=false)

# ── Support boundary lines ──
vline!(p, [phi_c], color=:blue, lw=1.5, ls=:dash, label=false)
al = range(0.50, 1.10, length=100)
plot!(p, al, (phi_c .- al .* q) ./ sq, color=:red, lw=1.5, ls=:dash, label=false)

# ── Boundary labels (direct, no legend) ──
annotate!(p, phi_c - 0.015, 0.48, text("φ₁=φ_c", FONT_ANN, :right, :blue))
annotate!(p, 0.58, 0.42, text("φ_μ=φ_c", FONT_ANN, :left, :red))

# ── Key points ──
scatter!(p, [a_ret], [b_ret], color=:green, markersize=8, markershape=:star5,
    markerstrokewidth=1, markerstrokecolor=:black, label=false)
scatter!(p, [a_cen], [b_cen], color=:orange, markersize=8, markershape=:diamond,
    markerstrokewidth=1, markerstrokecolor=:black, label=false)
scatter!(p, [a_ret], [b_entry], color=:red, markersize=6,
    markerstrokewidth=1, markerstrokecolor=:black, label=false)

# Point labels
annotate!(p, a_ret + 0.01, b_ret - 0.025,
    text("retrieval", FONT_ANN, :left, :green4))
annotate!(p, a_cen - 0.01, b_cen - 0.03,
    text("centroid", FONT_ANN, :right, :darkorange))
annotate!(p, a_ret + 0.01, b_entry + 0.02,
    text("entry", FONT_ANN, :left, :red3))

# ── Escape path ──
plot!(p, [a_ret, a_ret], [b_ret + 0.02, b_entry - 0.02],
    color=:black, lw=2.5, arrow=true, label=false)
plot!(p, [a_ret - 0.005, a_cen + 0.005], [b_entry, b_cen],
    color=:black, lw=2.5, arrow=true, label=false)

# Step labels
annotate!(p, a_ret - 0.015, (b_ret + b_entry) / 2,
    text("① ΔE=0", FONT_ANN, :right, :black))
annotate!(p, (a_ret + a_cen) / 2 + 0.015, (b_entry + b_cen) / 2 + 0.025,
    text("② ΔE<0", FONT_ANN, :left, :black))

# Energy values
E_ret = -(1/b_lsr) * log(max(0, 1-b_lsr+b_lsr*phi_eq))
E_cen = -(1/b_lsr) * log(2*max(0, 1-b_lsr+b_lsr*phi_cen))
annotate!(p, a_ret - 0.015, b_ret + 0.025,
    text(@sprintf("E/N=%.3f", E_ret), FONT_ANN-1, :right, :green4))
annotate!(p, a_cen + 0.01, b_cen + 0.025,
    text(@sprintf("E/N=%.3f", E_cen), FONT_ANN-1, :left, :darkorange))

# ── Region labels ──
annotate!(p, 0.96, 0.15, text("overlap", FONT_ANN-1, :center, :gray40))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "centroid_geometry.$ext"))
end
println("Saved: $(out_dir)/centroid_geometry.{png,pdf}")

#=
Centroid geometry for paper Figure 2 (left panel)
────────────────────────────────────────────────────────────────────────
Shows the (φ₁, v) plane with support caps, escape path, and centroid.
Uses paper notation: φ₁, v, φ_{1μ}, φ_c, v_entry, φ_cen.
Output: panels_paper/centroid_geometry.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using LaTeXStrings

# ──────────────── Parameters (paper values at α=0.20, T=0.325) ────────────────
const b_lsr  = 2 + sqrt(2)
const φ_c    = (b_lsr - 1) / b_lsr       # 0.707
const φ_1μ   = sqrt(1 - exp(-2*0.20))    # 0.574 (max inter-pattern overlap)
const sφ     = sqrt(1 - φ_1μ^2)          # sqrt(1 - φ_{1μ}²)

const φ_eq   = 0.90                       # single-pattern retrieval equilibrium
const φ_cen  = sqrt((1 + φ_1μ) / 2)      # geometric centroid (T=0)
const v_entry = (φ_c - φ_eq * φ_1μ) / sφ # entry value of v (Eq. 13)
const v_cen   = φ_cen * (1 - φ_1μ) / sφ  # v at centroid (φ₁=φ_μ=φ_cen)

# ──────────────── Figure settings (match paper panels) ────────────────
const FIG_DPI  = 300
const FIG_W    = round(Int, 86 / 25.4 * 100)   # 86mm → 339px
const FIG_H    = round(Int, FIG_W * 0.85)       # slightly shorter than square
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# ──────────────── Plot ────────────────

# Sphere boundary: φ₁² + v² = 1
θ = range(0, π/2, length=300)
circ_a = cos.(θ); circ_v = sin.(θ)

p = plot(circ_a, circ_v, color=:black, lw=1.5, label=false,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    xlabel=L"\phi_1", ylabel=L"v",
    xlims=(0.45, 1.03), ylims=(-0.02, 0.55),
    legend=false, aspect_ratio=:auto,
    left_margin=1Plots.mm, right_margin=1Plots.mm,
    top_margin=0Plots.mm, bottom_margin=1Plots.mm)

# ── Filled support regions ──

# ξ¹ support: φ₁ > φ_c (blue shading)
af = range(φ_c, 1.0, length=200)
vu = sqrt.(max.(0, 1.0 .- af.^2))
vu_clip = min.(vu, 0.55)
plot!(p, vcat(af, reverse(af)), vcat(vu_clip, zeros(length(af))),
    fillrange=0, fillalpha=0.12, fillcolor=:blue, lw=0, label=false)

# ξ^μ support: φ_μ = φ₁·φ_{1μ} + v·√(1-φ_{1μ}²) > φ_c
# i.e., v > (φ_c - φ₁·φ_{1μ}) / √(1-φ_{1μ}²)
a_f2 = Float64[]; v_lo2 = Float64[]; v_hi2 = Float64[]
for a in range(0.45, 1.0, length=400)
    vb = (φ_c - a * φ_1μ) / sφ      # boundary line
    vc = sqrt(max(0, 1.0 - a^2))      # sphere limit
    v_lo = max(vb, 0.0)
    v_hi = min(vc, 0.55)
    if v_lo < v_hi
        push!(a_f2, a); push!(v_lo2, v_lo); push!(v_hi2, v_hi)
    end
end
plot!(p, vcat(a_f2, reverse(a_f2)), vcat(v_hi2, reverse(v_lo2)),
    fillrange=0, fillalpha=0.10, fillcolor=:red, lw=0, label=false)

# ── Support boundary lines ──
vline!(p, [φ_c], color=:blue, lw=1.5, ls=:dash, label=false)
al = range(0.45, 1.03, length=100)
vl = (φ_c .- al .* φ_1μ) ./ sφ
plot!(p, al, vl, color=:red, lw=1.5, ls=:dash, label=false)

# ── Boundary labels ──
annotate!(p, φ_c - 0.015, 0.48,
    text(L"\phi_1 = \varphi_c", FONT_ANN, :right, :blue))
annotate!(p, 0.52, 0.38,
    text(L"\phi_\mu = \varphi_c", FONT_ANN, :left, :red))

# ── Key points ──
# Retrieval: (φ_eq, 0)
scatter!(p, [φ_eq], [0.0], color=:green, markersize=8, markershape=:star5,
    markerstrokewidth=1, markerstrokecolor=:black, label=false)
# Centroid: (φ_cen, v_cen)
scatter!(p, [φ_cen], [v_cen], color=:orange, markersize=7, markershape=:diamond,
    markerstrokewidth=1, markerstrokecolor=:black, label=false)
# Entry: (φ_eq, v_entry)
scatter!(p, [φ_eq], [v_entry], color=:red, markersize=5,
    markerstrokewidth=1, markerstrokecolor=:black, label=false)

# Point labels
annotate!(p, φ_eq + 0.01, -0.02,
    text(L"\phi_\mathrm{eq}", FONT_ANN, :left, :green4))
annotate!(p, φ_cen + 0.01, v_cen + 0.025,
    text(L"\varphi_\mathrm{cen}^{(0)}", FONT_ANN, :left, :darkorange))
annotate!(p, φ_eq + 0.01, v_entry + 0.02,
    text(L"v_\mathrm{entry}", FONT_ANN, :left, :red3))

# ── Escape path arrows ──
# Stage 1: vertical (φ_eq, 0) → (φ_eq, v_entry)
plot!(p, [φ_eq, φ_eq], [0.02, v_entry - 0.015],
    color=:black, lw=2.5, arrow=true, label=false)
# Stage 2: diagonal (φ_eq, v_entry) → (φ_cen, v_cen)
plot!(p, [φ_eq - 0.005, φ_cen + 0.005], [v_entry + 0.005, v_cen - 0.005],
    color=:black, lw=2.5, arrow=true, label=false)

# Step labels
annotate!(p, φ_eq - 0.02, v_entry / 2,
    text(L"\Delta E = 0", FONT_ANN, :right, :black))
annotate!(p, (φ_eq + φ_cen) / 2 + 0.02, (v_entry + v_cen) / 2 + 0.02,
    text(L"\Delta E < 0", FONT_ANN, :left, :black))

# ── Energy values at key points ──
E_ret = -(1/b_lsr) * log(max(1e-30, 1-b_lsr+b_lsr*φ_eq))
E_cen = -(1/b_lsr) * log(2*max(1e-30, 1-b_lsr+b_lsr*φ_cen))
annotate!(p, φ_eq + 0.01, 0.025,
    text(@sprintf("E/N = %.3f", E_ret), FONT_ANN-1, :left, :green4))
annotate!(p, φ_cen - 0.01, v_cen - 0.03,
    text(@sprintf("E/N = %.3f", E_cen), FONT_ANN-1, :right, :darkorange))

# ── Sphere label ──
annotate!(p, 0.82, 0.50,
    text(L"\phi_1^2 + v^2 = 1", FONT_ANN-1, :center, rotation=0, :gray40))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "centroid_geometry.$ext"))
end
println("Saved: panels_paper/centroid_geometry.{png,pdf}")
@printf("φ_{1μ} = %.3f, φ_eq = %.3f, φ_cen = %.3f\n", φ_1μ, φ_eq, φ_cen)
@printf("v_entry = %.3f, v_cen = %.3f\n", v_entry, v_cen)
@printf("E_ret/N = %.4f, E_cen/N = %.4f\n", E_ret, E_cen)

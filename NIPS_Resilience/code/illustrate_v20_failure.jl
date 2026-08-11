#=
2D schematic of v20's blind spot.
- ξ¹ at angle 0 (right). Chain x has wandered to θ_x.
- Live sector |θ| < θ_cut (blue cone) → top-K patterns kept in LSE.
- Outside the cone → "dropped" patterns averaged into the constant C.
- Highlight a single dropped pattern ξ_ν just outside the cone on the side
  the chain has wandered toward: low overlap with ξ¹ but high overlap with x.
  v20 cannot see it.

Output: ../panels_paper/illustrate_v20_dropped_escape.png
=#
ENV["GKSwstype"] = "100"
using Plots, Printf, Random

Random.seed!(7)

const θ_cut = π/8         # ≈22.5° — live cone around ξ¹
const θ_x   = π/4         # 45°   — chain has wandered here
const θ_ν   = θ_x + π/12  # 60°   — escape pattern, just outside cone, near chain

# Background patterns: uniform on circle, excluding a small arc around θ_ν
# so the explicit "escape pattern" stands out.
n_bg = 120
θ_bg = 2π .* rand(n_bg) .- π
keep = abs.(rem.(θ_bg .- θ_ν .+ π, 2π) .- π) .> π/24   # excise 7.5° around θ_ν
θ_bg = θ_bg[keep]

live_mask    = abs.(θ_bg) .< θ_cut
dropped_mask = .!live_mask

xξ, yξ = 1.0, 0.0
xx, yy = cos(θ_x), sin(θ_x)
xc, yc = cos(θ_ν), sin(θ_ν)

φ_x_ξ1   = cos(θ_x - 0)
φ_ν_ξ1   = cos(θ_ν - 0)
φ_ν_x    = cos(θ_ν - θ_x)
φ_cut_val = cos(θ_cut)

# ──────── plot ────────
default(guidefontsize=10, tickfontsize=8, legendfontsize=8)
p = plot(size=(780, 780), aspect_ratio=:equal, framestyle=:none,
         xlims=(-1.30, 1.45), ylims=(-1.30, 1.30),
         legend=:bottomleft, background_color=:white)

# unit circle
ts = range(0, 2π, length=400)
plot!(p, cos.(ts), sin.(ts), color=:black, lw=1, label="")

# live cone shading
fan_t = range(-θ_cut, θ_cut, length=80)
fan_x = vcat(0.0, cos.(fan_t), 0.0)
fan_y = vcat(0.0, sin.(fan_t), 0.0)
plot!(p, fan_x, fan_y, seriestype=:shape,
      fillcolor=:steelblue, fillalpha=0.10,
      linecolor=:steelblue, linealpha=0.4, label="")

# cone boundary rays (φ_cut)
plot!(p, [0, cos(θ_cut)],  [0, sin(θ_cut)],  color=:steelblue, lw=1, ls=:dot, label="")
plot!(p, [0, cos(-θ_cut)], [0, sin(-θ_cut)], color=:steelblue, lw=1, ls=:dot, label="")
annotate!(p, 0.55*cos(θ_cut)+0.02, 0.55*sin(θ_cut)+0.06,
          text(@sprintf("φ_cut = %.2f", φ_cut_val), :left, 9, :steelblue))

# patterns
scatter!(p, cos.(θ_bg[dropped_mask]), sin.(θ_bg[dropped_mask]),
         ms=4, mc=:gray70, msc=:gray60, markerstrokewidth=0,
         label="dropped (averaged into C)")
scatter!(p, cos.(θ_bg[live_mask]),    sin.(θ_bg[live_mask]),
         ms=5, mc=:steelblue, msc=:steelblue, markerstrokewidth=0,
         label="live (top-K, kept in LSE)")

# ξ¹
scatter!(p, [xξ], [yξ], ms=14, mc=:green, msc=:darkgreen,
         marker=:star5, label="ξ¹  (target)")
annotate!(p, xξ+0.08, yξ-0.04, text("ξ¹", :left, 11, :darkgreen))

# chain x  (orange diamond, with dashed radius from origin)
plot!(p, [0, xx], [0, yy], color=:orange, lw=1.5, ls=:dash, label="")
scatter!(p, [xx], [yy], ms=12, mc=:orange, msc=:darkorange,
         marker=:diamond, label="chain x")
annotate!(p, 1.10*xx-0.05, 1.10*yy+0.05,
          text("x", :left, 12, :darkorange))
annotate!(p, 0.45*xx-0.06, 0.45*yy-0.08,
          text(@sprintf("φ_x¹ = %.2f", φ_x_ξ1), :right, 9, :darkorange))

# escape pattern  (red triangle)
scatter!(p, [xc], [yc], ms=13, mc=:red, msc=:darkred,
         marker=:utriangle, label="ξ_ν  (closest dropped → escape route)")
plot!(p, [xx, xc], [yy, yc], color=:red, lw=2.5, label="")
annotate!(p, 1.10*xc, 1.10*yc+0.03,
          text("ξ_ν", :left, 12, :darkred))

# title with the two key overlaps
title!(p, @sprintf("φ(ξ_ν, ξ¹) = %.2f  <  φ_cut  →  dropped     |     φ(ξ_ν, x) = %.2f  →  dominates chain's local energy",
    φ_ν_ξ1, φ_ν_x), titlefontsize=10)

annotate!(p, -1.20, -1.20,
          text("2-D schematic of the N-sphere geometry  (high-N analogue: " *
               "perpendicular space has many directions)", :left, 8, :gray40))

out_dir = joinpath(@__DIR__, "..", "panels_paper"); mkpath(out_dir)
for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "illustrate_v20_dropped_escape.$ext"))
end
println("Saved: panels_paper/illustrate_v20_dropped_escape.{png,pdf}")
@printf("φ_x¹ = %.3f   φ_ν¹ = %.3f   φ_νx = %.3f   φ_cut = %.3f\n",
        φ_x_ξ1, φ_ν_ξ1, φ_ν_x, φ_cut_val)

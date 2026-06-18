#=
LO cusp φ_c(α) = α + g_max  vs  sample extreme
φ_max(α) = √(1 − e^{−2α}) of M = e^{αN} inter-pattern overlaps
under the spherical density.  β_net = 1.

Output: panels_paper/cusp_vs_extreme.{png,pdf,eps}
=#

using Plots
using LaTeXStrings

default(dpi = 450)   # Plots.jl/GR scales PNG by dpi/100; this yields >300 ppi
                     # at the printed 0.95*3.3 in width.

const φ_star = (sqrt(5) - 1)/2          # golden ratio: saddle at β=1
const g_max  = 0.5*log(1 - φ_star^2) + φ_star
const α_c    = 1.0 - g_max               # capacity at β=1

φ_max(α)   = sqrt(1 - exp(-2α))
φ_c(α)     = α + g_max          # = (α + g_max(β))/β at β = 1
φ_gauss(α) = sqrt(2α)
φ_grey(α)  = 1 - α              # 1 - α/β at β = 1

α_grid = collect(0.0:0.001:0.7)
clip1(v) = v > 1 ? NaN : v
y_max  = [φ_max(a)          for a in α_grid]
y_c    = [clip1(φ_c(a))     for a in α_grid]
y_g    = [clip1(φ_gauss(a)) for a in α_grid]
y_grey = [φ_grey(a)         for a in α_grid]

p = plot(size = (266, 177),
         xlabel = L"\alpha",
         ylabel = L"\varphi",
         legend = :bottomright,
         legendfontsize = 5,
         foreground_color_legend = nothing,
         background_color_legend = RGBA(1,1,1,0.75),
         framestyle = :box,
         grid = :on, gridalpha = 0.25,
         guidefontsize = 7, tickfontsize = 6,
         left_margin = 4Plots.mm, bottom_margin = 3Plots.mm,
         right_margin = 2Plots.mm, top_margin = 1Plots.mm)

plot!(p, α_grid, y_c;
      lw = 1.6, color = RGB(0.75, 0.10, 0.20),
      label = L"\varphi_c=\alpha+g_{\max}(\beta_{\rm net})")
plot!(p, α_grid, y_max;
      lw = 1.6, color = RGB(0.10, 0.20, 0.55),
      label = L"\varphi_{\max}=\sqrt{1-e^{-2\alpha}}")
plot!(p, α_grid, y_g;
      lw = 1.2, ls = :dot, color = RGB(0.10, 0.20, 0.55),
      label = L"\varphi=\sqrt{2\alpha}")
plot!(p, α_grid, y_grey;
      lw = 1.2, ls = :dash, color = :gray,
      label = L"\varphi=1-\alpha/\beta_{\rm net}")

xlims!(p, (0, 0.7))
ylims!(p, (0, 1.03))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
stem = "cusp_vs_extreme"
out_png = joinpath(outdir, stem * ".png")
out_pdf = joinpath(outdir, stem * ".pdf")
savefig(p, out_png); savefig(p, out_pdf)
run(`pdfcrop $out_pdf $out_pdf`)
println("Saved: ", out_png, "  ", out_pdf)

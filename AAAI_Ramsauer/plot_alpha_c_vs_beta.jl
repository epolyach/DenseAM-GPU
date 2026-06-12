#=
Zero-T capacity α_c^sd(β_net) of the LSE DAM vs Ramsauer's
β_net-independent prediction.

α_c^sd(β_net, 0) = β_net − g_max(β_net), with
  φ*(β_net) = (−1 + √(1 + 4β_net²)) / (2β_net),
  g_max(β_net) = ½ log(1 − φ*²) + β_net · φ*.

Output: panels_paper/alpha_c_vs_beta.{png,pdf,eps}
=#

using Plots
using LaTeXStrings

phi_star(β) = (-1 + sqrt(1 + 4β^2)) / (2β)
function g_max(β)
    φ = phi_star(β)
    0.5 * log(1 - φ^2) + β * φ
end
α_sd(β) = β - g_max(β)

# log-spaced grid 0.01 .. 100
βs = 10 .^ range(-2, stop = 2, length = 401)
α_sd_curve = [α_sd(β) for β in βs]

# 90 mm wide → 255 pt; aspect ~ 1.5:1
p = plot(size = (255, 170),
         xlabel = L"\beta_{\mathrm{net}}",
         ylabel = L"\alpha_c^{\mathrm{sd}}",
         xscale = :log10,
         xticks = ([0.01, 0.1, 1.0, 10.0, 100.0],
                   ["0.01", "0.1", "1", "10", "100"]),
         legend = false,
         framestyle = :box,
         grid = :on, gridalpha = 0.25,
         guidefontsize = 7, tickfontsize = 6,
         left_margin = 4Plots.mm, bottom_margin = 3Plots.mm,
         right_margin = 2Plots.mm, top_margin = 1Plots.mm)

plot!(p, βs, α_sd_curve;
      lw = 1.6, color = RGB(0.75, 0.10, 0.20))

hline!(p, [0.5];
       lw = 1.2, ls = :dash, color = RGB(0.10, 0.20, 0.55))

xlims!(p, (1e-2, 1e2))
ylims!(p, (-0.05, 3.0))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
stem = "alpha_c_vs_beta"
out_png = joinpath(outdir, stem * ".png")
out_pdf = joinpath(outdir, stem * ".pdf")
out_eps = joinpath(outdir, stem * ".eps")
savefig(p, out_png); savefig(p, out_pdf)
run(`pdftops -eps $out_pdf $out_eps`)
println("Saved: ", out_png, "  ", out_pdf, "  ", out_eps)

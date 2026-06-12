#=
LO free energy F(φ_1)/N at α = 0.5, β_net = 1, for several T.

Output: panels_paper/cusp_illustration.{png,pdf,eps}
=#

using Plots
using LaTeXStrings
using Printf

const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(φ_star) + φ_star
const α      = 0.50
const cusp   = α + g_max

phi_eq(T) = 0.5*(-T + sqrt(T^2 + 4))
T_sp_LO(α) = (1 - (α + g_max)^2)/(α + g_max)
F_per_N(φ, T) = -max(φ, cusp) - 0.5*T*log(1 - φ^2)

const Tsp = T_sp_LO(α)

const Ts     = [0.05, 0.15, 0.25, Tsp, 0.35]
const labels = [L"T = 0.05",
                L"T = 0.15",
                L"T = 0.25",
                L"T = T_{\mathrm{sp}}",
                L"T = 0.35"]
const cols   = [RGB(0.10, 0.20, 0.55),
                RGB(0.20, 0.45, 0.75),
                RGB(0.40, 0.65, 0.85),
                RGB(0.80, 0.20, 0.10),
                RGB(0.55, 0.55, 0.55)]

φ_grid = collect(0.55:0.001:0.999)

p = plot(size = (255, 170),
         xlabel = L"\varphi_1",
         ylabel = L"F(\varphi_1)/N",
         legend = :topleft,
         legendfontsize = 5,
         foreground_color_legend = nothing,
         background_color_legend = RGBA(1,1,1,0.75),
         framestyle = :box,
         grid = :on, gridalpha = 0.25,
         guidefontsize = 7, tickfontsize = 6,
         left_margin = 4Plots.mm, bottom_margin = 3Plots.mm,
         right_margin = 2Plots.mm, top_margin = 1Plots.mm)

for (T, lab, col) in zip(Ts, labels, cols)
    Fs = [F_per_N(φ, T) for φ in φ_grid]
    plot!(p, φ_grid, Fs; lw = 1.4, color = col, label = lab)
    φe = phi_eq(T)
    if φe > cusp
        scatter!(p, [φe], [F_per_N(φe, T)];
                 markershape = :circle, markersize = 3,
                 markercolor = col, markerstrokewidth = 0,
                 label = "")
    end
end

vline!(p, [cusp]; lw = 0.8, ls = :dash, color = :black, alpha = 0.55, label = "cusp")

xlims!(p, (0.55, 1.0))
ylims!(p, (-1.07, -0.45))

annotate!(p, 0.715, -0.985, text("Saddle valley", 5, :center, :black))
annotate!(p, 0.94, -1.005, text("Retrieval\nbasin", 5, :center, :black))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "cusp_illustration.png")
out_pdf = joinpath(outdir, "cusp_illustration.pdf")
out_eps = joinpath(outdir, "cusp_illustration.eps")
savefig(p, out_png); savefig(p, out_pdf)
run(`pdftops -eps $out_pdf $out_eps`)
println("Saved: ", out_png, "  ", out_pdf, "  ", out_eps)

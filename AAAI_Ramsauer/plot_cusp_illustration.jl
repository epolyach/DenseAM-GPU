#=
LO free energy f(φ) at α = 0.5, β_net = 1, for several T.

Output: panels_paper/cusp_illustration.{png,pdf}
Styling matches plot_cusp_vs_extreme.jl: dpi 450, source 266×177 pt,
pdfcrop to ≈226×144 pt = 3.139×2.00 in (= 0.95×3.3 in).
=#

using Plots
using LaTeXStrings
using Printf

default(dpi = 450)

const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(1 - φ_star^2) + φ_star          # g_max(β=1)
const α      = 0.50
const cusp   = α + g_max

phi_eq(T)  = 0.5*(-T + sqrt(T^2 + 4))
T_sp_LO(α) = (1 - (α + g_max)^2)/(α + g_max)
f_LO(φ, T) = -max(φ, cusp) - 0.5*T*log(1 - φ^2)

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

p = plot(size = (259, 177),
         xlabel = L"\varphi",
         ylabel = L"f(\varphi)",
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
    fs = [f_LO(φ, T) for φ in φ_grid]
    plot!(p, φ_grid, fs; lw = 1.4, color = col, label = lab)
    φe = phi_eq(T)
    if φe > cusp
        scatter!(p, [φe], [f_LO(φe, T)];
                 markershape = :circle, markersize = 3,
                 markercolor = col, markerstrokewidth = 0,
                 label = "")
    end
end

vline!(p, [cusp]; lw = 1.0, ls = :dash, color = :black, alpha = 0.55, label = "cusp")

xlims!(p, (0.55, 1.0))
ylims!(p, (-1.07, -0.45))

annotate!(p, 0.71, -0.98, text("SV", 6, :center, :black))
annotate!(p, 0.94, -0.98, text("RB", 6, :center, :black))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "cusp_illustration.png")
out_pdf = joinpath(outdir, "cusp_illustration.pdf")
savefig(p, out_png); savefig(p, out_pdf)
run(`pdfcrop $out_pdf $out_pdf`)
println("Saved: ", out_png, "  ", out_pdf)

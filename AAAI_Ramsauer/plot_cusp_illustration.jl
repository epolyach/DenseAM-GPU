#=
Illustrate the cusp in the leading-order free energy F(φ_1)/N at fixed α
and several T.  As T rises, the basin minimum φ_eq(T) slides toward the
cusp at φ_1 = α + g_max.  At T = T_sp(α), the minimum reaches the cusp
and ceases to exist; that is the single-pattern spinodal.

Output: panels_paper/cusp_illustration.{png,pdf}
=#

using Plots
using Printf

const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(φ_star) + φ_star          # ≈ 0.3774
const α      = 0.50
const cusp   = α + g_max                          # ≈ 0.8774

phi_eq(T) = 0.5*(-T + sqrt(T^2 + 4))
T_sp_LO(α) = (1 - (α + g_max)^2)/(α + g_max)
F_per_N(φ, T) = -max(φ, cusp) - 0.5*T*log(1 - φ^2)

const Tsp = T_sp_LO(α)                            # ≈ 0.262

const Ts     = [0.05, 0.15, 0.25, Tsp, 0.35]
const labels = ["T = 0.05",
                "T = 0.15",
                "T = 0.25",
                @sprintf("T = T_sp ≈ %.3f", Tsp),
                "T = 0.35 (basin gone)"]
const cols   = [RGB(0.10, 0.20, 0.55),
                RGB(0.20, 0.45, 0.75),
                RGB(0.40, 0.65, 0.85),
                RGB(0.80, 0.20, 0.10),
                RGB(0.55, 0.55, 0.55)]

φ_grid = collect(0.55:0.001:0.999)

p = plot(size=(820, 520),
         xlabel = "chain–target overlap  φ_1",
         ylabel = "free energy per spin  F(φ_1)/N",
         legend = :topleft,
         framestyle = :box,
         grid = :on, gridalpha = 0.25,
         guidefontsize = 11, tickfontsize = 10)

for (T, lab, col) in zip(Ts, labels, cols)
    Fs = [F_per_N(φ, T) for φ in φ_grid]
    plot!(p, φ_grid, Fs, lw = 2.2, color = col, label = lab)
    # Basin minimum (only if it exists, i.e. φ_eq(T) > cusp)
    φe = phi_eq(T)
    if φe > cusp
        scatter!(p, [φe], [F_per_N(φe, T)],
                 markershape = :circle, markersize = 6,
                 markercolor = col, markerstrokewidth = 0,
                 label = "")
    end
end

# Mark cusp
vline!(p, [cusp], lw = 1.6, ls = :dash, color = :black, alpha = 0.6,
       label = @sprintf("cusp at φ = α + g_max = %.4f", cusp))

# Annotations
annotate!(p, cusp - 0.05, -0.55, text("noise floor\n(slope 0)", 9, :right, :black))
annotate!(p, 0.96, -0.96, text("retrieval branch\n(slope -1)", 9, :right, :black))

# Save
outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "cusp_illustration.png")
out_pdf = joinpath(outdir, "cusp_illustration.pdf")
out_eps = joinpath(outdir, "cusp_illustration.eps")
savefig(p, out_png); savefig(p, out_pdf)
run(`pdftops -eps $out_pdf $out_eps`)
println("Saved:")
println("  ", out_png)
println("  ", out_pdf)
println("  ", out_eps)

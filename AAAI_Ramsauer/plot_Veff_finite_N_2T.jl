#=
Compare V_eff(φ) at two temperatures T = 0.005 (solid) and T = 0.003 (dotted).
Three loads α = 0.62, 0.65, 0.70, finite N from the M = 4.4e6 Nramp.
Shows how the residual basin moves and deepens as T decreases.

Each panel includes:
  - LO landscape (N → ∞):   black, solid at T = 0.005, dotted at T = 0.003.
  - Finite-N landscape V_N: red,   solid at T = 0.005, dotted at T = 0.003.
  - Red dots at the V_N residual-basin minima.
  - Vertical grey lines at φ_eq(T) for both T (solid and dotted).
  - Vertical grey line at φ_c (only when φ_c < 1).

Outputs panels_paper/Veff_finite_N_2T.{png,pdf}.
=#

using Printf
using Plots
using LaTeXStrings

default(dpi = 600)

const φ_star = (sqrt(5)-1)/2
const g_max  = 0.5*log(1 - φ_star^2) + φ_star
const C_pre  = 1/sqrt(1 - φ_star^4)
const lnC    = log(C_pre)
const M_DATA = 4.4e6
const T1     = 0.005
const T2     = 0.003
const β      = 1.0

φ_c(α) = α + g_max

V_LO(φ, α, T) = -max(φ, φ_c(α)) - (T/2)*log(1 - φ^2)

function V_N(φ, α, T, N)
    a = β*N*φ
    b = N*φ_c(α) + lnC
    m = max(a, b)
    E = -(m + log(1 + exp(-abs(a-b)))) / (β*N)
    return E - (T*(N-3)/(2N))*log(1 - φ^2)
end

N_for(α) = max(12, round(Int, log(M_DATA)/α))

αs = (0.62, 0.65, 0.70)
labels = [L"\alpha=0.62,\ N=%$(N_for(0.62))",
          L"\alpha=0.65,\ N=%$(N_for(0.65))",
          L"\alpha=0.70,\ N=%$(N_for(0.70))"]

# φ_eq(T) for both temperatures.
φ_eq1 = 0.5*(sqrt(T1^2 + 4) - T1)
φ_eq2 = 0.5*(sqrt(T2^2 + 4) - T2)
v_eq1 = -log10(1 - φ_eq1)
v_eq2 = -log10(1 - φ_eq2)

v_grid = collect(0.0:0.005:3.0)
φ_grid = 1 .- 10 .^ (-v_grid)

xtick_v     = [0.0, 1.0, 2.0, 3.0]
xtick_label = ["0", "0.9", "0.99", "0.999"]

plots = []
for (i, (α, lbl)) in enumerate(zip(αs, labels))
    N = N_for(α)

    V_lo1 = V_LO.(φ_grid, α, T1)
    V_lo2 = V_LO.(φ_grid, α, T2)
    V_N1  = V_N.(φ_grid, α, T1, N)
    V_N2  = V_N.(φ_grid, α, T2, N)

    v_c = φ_c(α) < 1 ? -log10(1 - φ_c(α)) : nothing

    # Y-range: fit both T curves' minima and a sensible top.
    Vmin = minimum([minimum(V_N1), minimum(V_N2)])
    Vmax = maximum([maximum(V_lo1), maximum(V_lo2)])
    span = Vmax - Vmin
    ylo  = Vmin - 0.08*span
    yhi  = Vmax + 0.30*span

    ylabel_str = i == 1 ? L"V_N(\phi) / N" : ""

    # Base plot: LO at T1 (solid black).
    p = plot(v_grid, V_lo1;
             color = :black, lw = 1.0, ls = :solid,
             label = L"\mathrm{LO},\ T=%$T1",
             xlabel = L"\phi",
             ylabel = ylabel_str,
             yformatter = x -> @sprintf("%.3f", x),
             xticks = (xtick_v, xtick_label),
             title = lbl, titlefontsize = 6,
             guidefontsize = 7, tickfontsize = 5,
             framestyle = :box, legend = :topleft, legendfontsize = 4,
             foreground_color_legend = nothing,
             background_color_legend = RGBA(1,1,1,0.85),
             left_margin   = (i == 1 ? 5Plots.mm : 0Plots.mm),
             bottom_margin = 2Plots.mm,
             top_margin    = 0Plots.mm,
             right_margin  = (i == 3 ? 3Plots.mm : 1Plots.mm),
             aspect_ratio  = :auto)

    # LO at T2 (dotted black).
    plot!(p, v_grid, V_lo2;
          color = :black, lw = 1.0, ls = :dot,
          label = L"\mathrm{LO},\ T=%$T2")

    # Finite-N at T1 (solid red).
    plot!(p, v_grid, V_N1;
          color = :red, lw = 1.0, ls = :solid,
          label = L"V_N,\ T=%$T1")

    # Finite-N at T2 (dotted red).
    plot!(p, v_grid, V_N2;
          color = :red, lw = 1.0, ls = :dot,
          label = L"V_N,\ T=%$T2")

    # φ_c vertical (only if on the sphere).
    if v_c !== nothing && v_c <= 3
        vline!(p, [v_c]; color = :grey, ls = :dashdot, lw = 0.8, label = L"\varphi_c")
    end

    # φ_eq vertical lines (solid and dotted, grey).
    vline!(p, [v_eq1]; color = :grey, ls = :solid, lw = 0.6,
           label = L"\varphi_{\mathrm{eq}}(T=%$T1)")
    vline!(p, [v_eq2]; color = :grey, ls = :dot,   lw = 0.6,
           label = L"\varphi_{\mathrm{eq}}(T=%$T2)")

    # Red dots at V_N residual-basin minima (search in v ∈ [1, 3]).
    mask_res = (v_grid .>= 1.0) .& (v_grid .<= 3.0)
    idx_res  = findall(mask_res)
    m1 = idx_res[argmin(V_N1[idx_res])]
    m2 = idx_res[argmin(V_N2[idx_res])]
    scatter!(p, [v_grid[m1]], [V_N1[m1]];
             color = :red, markersize = 3.0, markerstrokewidth = 0,
             markershape = :circle, label = "")
    scatter!(p, [v_grid[m2]], [V_N2[m2]];
             color = :red, markersize = 3.0, markerstrokewidth = 0,
             markershape = :diamond, label = "")

    # Black dots at LO retrieval-basin minima at both T (only when φ_eq > φ_c).
    if φ_eq1 > φ_c(α)
        scatter!(p, [v_eq1], [V_LO(φ_eq1, α, T1)];
                 color = :black, markersize = 3.0, markerstrokewidth = 0,
                 markershape = :circle, label = "")
    end
    if φ_eq2 > φ_c(α)
        scatter!(p, [v_eq2], [V_LO(φ_eq2, α, T2)];
                 color = :black, markersize = 3.0, markerstrokewidth = 0,
                 markershape = :diamond, label = "")
    end

    ylims!(p, (ylo, yhi))
    xlims!(p, (0, 3))
    push!(plots, p)
end

P = plot(plots...; layout = (1, 3), size = (469, 166))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "Veff_finite_N_2T.png")
out_pdf = joinpath(outdir, "Veff_finite_N_2T.pdf")
savefig(P, out_png)
savefig(P, out_pdf)
run(`pdfcrop $out_pdf $out_pdf`)
println("Saved: ", out_png)
println("       ", out_pdf)

println()
println("Local extrema of V_N at both temperatures:")
for (α, lbl) in zip(αs, labels)
    N = N_for(α)
    for T in (T1, T2)
        V = [V_N(φ, α, T, N) for φ in φ_grid]
        dV = diff(V)
        crit = Int[]
        for i in 1:length(dV)-1
            if sign(dV[i]) != sign(dV[i+1])
                push!(crit, i+1)
            end
        end
        @printf("  α=%.2f T=%.4f N=%d critical φ = %s; V = %s\n", α, T, N,
                join([@sprintf("%.4f", φ_grid[i]) for i in crit], ", "),
                join([@sprintf("%.5f", V[i]) for i in crit], ", "))
    end
end

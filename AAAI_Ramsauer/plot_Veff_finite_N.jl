#=
Plot V_eff(φ_1) at LO (N → ∞) and at finite N for three loads:
  α = 0.62 (right at the T=0 boundary)
  α = 0.65
  α = 0.70
all at T = 0.005, β = 1.

Each panel shows:
  - LO landscape: V_LO/N = -max(φ_1, φ_c) - (T/2) log(1-φ_1²)
  - Finite-N landscape: V_N/N = -(1/N) log(e^{Nφ_1} + e^{Nφ_c}) - (T/2) log(1-φ_1²)
    with the N(α) ramp used in the M=4.4e6 dataset.

Outputs panels_paper/Veff_finite_N.{png,pdf}.
=#

using Printf
using Plots
using LaTeXStrings

default(dpi = 300)

const φ_star = (sqrt(5)-1)/2
const g_max  = 0.5*log(1 - φ_star^2) + φ_star            # ≈ 0.3774
const C_pre  = 1/sqrt(1 - φ_star^4)                       # ≈ 1.082, Appendix A prefactor
const lnC    = log(C_pre)
const M_DATA = 4.4e6
const T_LO   = 0.005
const β      = 1.0

φ_c(α) = α + g_max

V_LO(φ, α, T) = 1 - max(φ, φ_c(α)) - (T/2)*log(1 - φ^2)

# Use logsumexp form to avoid overflow for large N. Includes Appendix A prefactor C_pre on the spurious branch and the exact spherical entropy (N-3)/(2N) (not the large-N limit 1/2).
function V_N(φ, α, T, N)
    a = β*N*φ
    b = N*φ_c(α) + lnC
    m = max(a, b)
    E = -(m + log(1 + exp(-abs(a-b)))) / (β*N)
    return 1 + E - (T*(N-3)/(2N))*log(1 - φ^2)
end

# Match Nramp: N(α) = round(log M / α), floored at 12
N_for(α) = max(12, round(Int, log(M_DATA)/α))

αs = (0.62, 0.65, 0.70)
labels = [L"\alpha=0.62,\ N=%$(N_for(0.62))",
          L"\alpha=0.65,\ N=%$(N_for(0.65))",
          L"\alpha=0.70,\ N=%$(N_for(0.70))"]

# Plot against v = -log10(1 - φ) so high φ (basin) is on the right.
# v = 0 → φ = 0; v = 1 → φ = 0.9; v = 2 → φ = 0.99; v = 3 → φ = 0.999.
v_grid = collect(0.0:0.005:3.0)
φ_grid = 1 .- 10 .^ (-v_grid)

# Manual φ-ticks at v = 0, 1, 2, 3.
xtick_v     = [0.0, 1.0, 2.0, 3.0]
xtick_label = ["0", "0.9", "0.99", "0.999"]

plots = []
for (i, (α, lbl)) in enumerate(zip(αs, labels))
    N = N_for(α)
    V_lo_curve = V_LO.(φ_grid, α, T_LO)
    V_N_curve  = V_N.(φ_grid, α, T_LO, N)

    # φ_c position on the v-axis (only when φ_c < 1)
    v_c = φ_c(α) < 1 ? -log10(1 - φ_c(α)) : nothing

    ylabel_str = i == 1 ? L"V_N(\varphi) / N" : ""

    p = plot(v_grid, V_lo_curve;
             color = :black, lw = 1.0, ls = :solid,
             label = L"\mathrm{LO}\ (N\to\infty)",
             xlabel = L"\varphi",
             ylabel = ylabel_str,
             yformatter = x -> @sprintf("%.3f", x),
             xticks = (xtick_v, xtick_label),
             title = lbl, titlefontsize = 6,
             guidefontsize = 7, tickfontsize = 5,
             framestyle = :box, legend = :topleft, legendfontsize = 4,
             foreground_color_legend = nothing,
             background_color_legend = RGBA(1,1,1,0.85),
             left_margin   = (i == 1 ? 4Plots.mm : 0Plots.mm),
             bottom_margin = 2Plots.mm,
             top_margin    = 0Plots.mm,
             right_margin  = (i == 3 ? 3Plots.mm : 1Plots.mm),
             aspect_ratio  = :auto)
    plot!(p, v_grid, V_N_curve;
          color = :red, lw = 1.0, ls = :solid,
          label = L"\mathrm{finite}\ N = %$N")
    if v_c !== nothing && v_c <= 3
        vline!(p, [v_c]; color = :grey, ls = :dot, lw = 0.8,
               label = L"\varphi_c")
    end

    # Red dot at V_N residual-basin minimum (search in φ ∈ [0.9, 0.999], i.e. v ∈ [1, 3]).
    mask_res     = (v_grid .>= 1.0) .& (v_grid .<= 3.0)
    idx_res      = findall(mask_res)
    min_res_idx  = idx_res[argmin(V_N_curve[idx_res])]
    scatter!(p, [v_grid[min_res_idx]], [V_N_curve[min_res_idx]];
             color = :red, markersize = 2.5, markerstrokewidth = 0, label = "")

    # Black dot at LO retrieval-basin minimum in the leftmost panel only.
    # LO basin exists only when φ_eq(T) > φ_c (here only at α = 0.62).
    if i == 1
        φ_eq = 0.5*(sqrt(T_LO^2 + 4) - T_LO)
        if φ_eq > φ_c(α)
            v_eq    = -log10(1 - φ_eq)
            V_LO_eq = V_LO(φ_eq, α, T_LO)
            scatter!(p, [v_eq], [V_LO_eq];
                     color = :black, markersize = 2.5, markerstrokewidth = 0, label = "")
        end
    end
    # Y-window: extra headroom at the top so the top-left legend has space
    # above the LO curve and doesn't overlap any line.
    Vmin = min(minimum(V_lo_curve), minimum(V_N_curve))
    Vmax = max(maximum(V_lo_curve), maximum(V_N_curve))
    span = Vmax - Vmin
    ylims!(p, (Vmin - 0.05*span, Vmax + 0.45*span))
    xlims!(p, (0, 3))
    push!(plots, p)
end

# Source size = 1.05 × target (6.2 in × 72 = 446 pt), so 469 pt wide.
# Height scaled by the same factor: 158 pt → 166 pt. After pdfcrop the
# output comes down to ≈ 6.2 in wide; LaTeX includes at width=6.2in.
P = plot(plots...; layout = (1, 3), size = (469, 166))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "Veff_finite_N.png")
out_pdf = joinpath(outdir, "Veff_finite_N.pdf")
savefig(P, out_png)
savefig(P, out_pdf)
# Trim margins fully (matches the convention used for other AAAI panels)
run(`pdfcrop $out_pdf $out_pdf`)
println("Saved: ", out_png)
println("       ", out_pdf)

# Report: where does V_N have a local minimum?
println()
println("Local extrema of V_N (numerical):")
for (α, lbl) in zip(αs, labels)
    N = N_for(α)
    V_vals = V_N.(φ_grid, α, T_LO, N)
    # Find sign changes of derivative
    dV = diff(V_vals)
    signs = sign.(dV)
    crit = Int[]
    for i in 1:length(signs)-1
        if signs[i] != signs[i+1]
            push!(crit, i+1)
        end
    end
    @printf("  %s: critical φ = %s; V at those = %s\n", lbl,
            join([@sprintf("%.4f", φ_grid[i]) for i in crit], ", "),
            join([@sprintf("%.5f", V_vals[i]) for i in crit], ", "))
    @printf("    V_N(0)   = %.5f,  V_N(0.99) = %.5f\n",
            V_N(0.0, α, T_LO, N), V_N(0.99, α, T_LO, N))
end

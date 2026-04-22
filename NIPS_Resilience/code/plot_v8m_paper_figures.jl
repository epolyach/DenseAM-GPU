#=
Publication figures from v8m LSR data (gpu2, no gaps)
────────────────────────────────────────────────────────────────────────
Output: panels_paper/heatmap_LSR.{png,pdf}
        panels_paper/cdf_LSR.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using Statistics

# ──────────────── Figure settings ────────────────
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)   # 86mm
const FIG_H = FIG_W
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7
const FONT_LEG   = 6

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = "panels_paper"
mkpath(out_dir)

# ──────────────── Read data ────────────────
println("Reading v8m data...")
lines = readlines(expanduser("~/Downloads/basin_stab_LSR_v8m.csv"))
n = length(lines) - 1
alpha = zeros(n); T = zeros(n); phi_a = zeros(n); phi_b = zeros(n)
phimax = zeros(n)
for i in 1:n
    f = split(lines[i+1], ",")
    alpha[i] = parse(Float64, f[1]); T[i] = parse(Float64, f[2])
    phi_a[i] = parse(Float64, f[4]); phi_b[i] = parse(Float64, f[5])
    phimax[i] = parse(Float64, f[7])
end

alphas = sort(unique(round.(alpha, digits=4)))
Ts = sort(unique(round.(T, digits=4)))
na = length(alphas); nT = length(Ts)

# Build grids
phi_grid = zeros(nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 0.001) .& (abs.(T .- Ts[iT]) .< 0.001)
    phi_grid[iT, ia] = mean(vcat(phi_a[mask], phi_b[mask]))
end

# ──────────────── LSR theory ────────────────
const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr

function φ_LSR(T)
    a = b_lsr*T + 1; bc = -(2 + T + T*b_lsr); c = T
    disc = bc^2 - 4*a*c; disc < 0 && return NaN
    return 1 - (-bc - sqrt(disc)) / (2*a)
end

u_LSR(φ) = -log(1 - b_lsr*(1-φ)) / b_lsr
s_func(φ) = 0.5 * log(1 - φ^2)
f_ret_LSR(T) = let φ = φ_LSR(T); u_LSR(φ) - T*s_func(φ); end

α_th_gauss = φ_c^2 / 2
α_c_gauss(T) = 0.5 * (1 - f_ret_LSR(T))^2
α_c_exact(T) = let fr = f_ret_LSR(T); arg = fr*(2-fr); arg <= 0 ? Inf : -0.5*log(arg); end
α_th_exact = -0.5 * log(1 - φ_c^2)

# ──────────────── HEATMAP ────────────────
println("Plotting heatmap...")

p1 = heatmap(alphas, Ts, phi_grid,
    color=:RdBu, clims=(0, 1),
    xlabel="α", ylabel="T",
    xlims=(0, 0.55), ylims=(0, 2.0),
    colorbar_title="⟨φ⟩",
    size=(FIG_W + 40, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, bottom_margin=0Plots.mm)

# Theory overlays — only physical branch for LSR
T_th = range(0.005, 2.0, length=500)
T_phys = [t for t in T_th if f_ret_LSR(t) <= 1.0]

# Gaussian
T_solid_g = [t for t in T_phys if α_c_gauss(t) >= α_th_gauss && α_c_gauss(t) <= 0.55]
α_solid_g = [α_c_gauss(t) for t in T_solid_g]
T_dot_g = [t for t in T_phys if α_c_gauss(t) < α_th_gauss]
α_dot_g = [α_c_gauss(t) for t in T_dot_g]
T_join_g = isempty(T_solid_g) ? 0.0 : maximum(T_solid_g)
plot!(p1, [α_th_gauss, α_th_gauss], [T_join_g, 2.0], color=:blue, lw=2, label=false)
!isempty(T_solid_g) && plot!(p1, α_solid_g, T_solid_g, color=:blue, lw=2, label="Gaussian")
!isempty(T_dot_g) && plot!(p1, α_dot_g, T_dot_g, color=:blue, lw=1.5, ls=:dot, label=false)

# Exact
T_solid_e = [t for t in T_phys if α_c_exact(t) >= α_th_exact && α_c_exact(t) <= 0.55]
α_solid_e = [α_c_exact(t) for t in T_solid_e]
T_dot_e = [t for t in T_phys if α_c_exact(t) < α_th_exact]
α_dot_e = [min(α_c_exact(t), 0.55) for t in T_dot_e]
T_join_e = isempty(T_solid_e) ? 0.0 : maximum(T_solid_e)
plot!(p1, [α_th_exact, α_th_exact], [T_join_e, 2.0], color=:red, lw=2, ls=:dash, label=false)
!isempty(T_solid_e) && plot!(p1, α_solid_e, T_solid_e, color=:red, lw=2, ls=:dash, label="Exact")
!isempty(T_dot_e) && plot!(p1, α_dot_e, T_dot_e, color=:red, lw=1.5, ls=:dot, label=false)

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "heatmap_LSR.$ext"))
end
println("Saved heatmap.")

# ──────────────── CDFs at selected (α, T) points ────────────────
println("Plotting CDFs...")

# Select interesting points spanning the transition
cdf_points = [
    (0.10, 0.325, "α=0.10 (deep retrieval)"),
    (0.20, 0.325, "α=0.20 (onset)"),
    (0.22, 0.325, "α=0.22"),
    (0.25, 0.325, "α=0.25 (≈α_th)"),
    (0.28, 0.325, "α=0.28"),
    (0.30, 0.325, "α=0.30"),
    (0.35, 0.325, "α=0.35"),
]

colors_cdf = [:darkblue, :blue, :dodgerblue, :green, :orange, :red, :darkred]

p2 = plot(xlabel="φ", ylabel="CDF",
    xlims=(-0.1, 1.05), ylims=(0, 1),
    legend=:topleft, legendfontsize=FONT_LEG,
    background_color_legend=RGBA(0.95,0.95,0.95,0.8),
    foreground_color_legend=RGBA(0.7,0.7,0.7,0.5),
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, bottom_margin=0Plots.mm)

# φ_c line
vline!(p2, [φ_c], color=:gray, lw=1, ls=:dash, label=false)

for (ci, (α_sel, T_sel, lbl)) in enumerate(cdf_points)
    mask = (abs.(alpha .- α_sel) .< 0.003) .& (abs.(T .- T_sel) .< 0.003)
    vals = sort(vcat(phi_a[mask], phi_b[mask]))
    isempty(vals) && continue
    cdf_y = range(0, 1, length=length(vals))
    plot!(p2, vals, cdf_y, color=colors_cdf[ci], lw=1.5, label=lbl)
end

for ext in ("png", "pdf")
    savefig(p2, joinpath(out_dir, "cdf_LSR_T0325.$ext"))
end
println("Saved CDF at T=0.325.")

# CDFs at fixed α, varying T
cdf_T_points = [
    (0.22, 0.125, "T=0.125"),
    (0.22, 0.225, "T=0.225"),
    (0.22, 0.325, "T=0.325"),
    (0.22, 0.525, "T=0.525"),
    (0.22, 0.825, "T=0.825"),
    (0.22, 1.025, "T=1.025"),
]

colors_T = [:darkblue, :blue, :green, :orange, :red, :darkred]

p3 = plot(xlabel="φ", ylabel="CDF",
    xlims=(-0.1, 1.05), ylims=(0, 1),
    legend=:topleft, legendfontsize=FONT_LEG,
    background_color_legend=RGBA(0.95,0.95,0.95,0.8),
    foreground_color_legend=RGBA(0.7,0.7,0.7,0.5),
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=0Plots.mm, bottom_margin=0Plots.mm)

vline!(p3, [φ_c], color=:gray, lw=1, ls=:dash, label=false)

for (ci, (α_sel, T_sel, lbl)) in enumerate(cdf_T_points)
    mask = (abs.(alpha .- α_sel) .< 0.003) .& (abs.(T .- T_sel) .< 0.003)
    vals = sort(vcat(phi_a[mask], phi_b[mask]))
    isempty(vals) && continue
    cdf_y = range(0, 1, length=length(vals))
    plot!(p3, vals, cdf_y, color=colors_T[ci], lw=1.5, label=lbl)
end

for ext in ("png", "pdf")
    savefig(p3, joinpath(out_dir, "cdf_LSR_alpha022.$ext"))
end
println("Saved CDF at α=0.22.")

# ──────────────── Print statistics for the interesting points ────────────────
println("\nDetailed statistics at T = 0.325:")
@printf("  %-5s  %4s  %5s  %6s  %6s  %6s  %6s  %6s\n",
    "α", "N", "n_dis", "⟨φ⟩", "σ(φ)", "⟨φ_max⟩", "frac<0.86", "frac<0.5")
println("  " * "─"^60)
for α_sel in [0.10, 0.15, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.35]
    T_sel = 0.325
    mask = (abs.(alpha .- α_sel) .< 0.003) .& (abs.(T .- T_sel) .< 0.003)
    pa = phi_a[mask]; pb = phi_b[mask]; pm = phimax[mask]
    vals = vcat(pa, pb)
    isempty(vals) && continue
    M = round(Int, exp(α_sel * 50))  # approximate
    N = round(Int, log(20000) / α_sel)  # approximate from v8m
    ndis = length(pa)
    @printf("  %.2f   %3d   %4d  %6.3f  %5.3f  %6.3f    %5.3f     %5.3f\n",
        α_sel, N, ndis, mean(vals), std(vals), mean(pm),
        mean(vals .< 0.86), mean(vals .< 0.5))
end

println("\nDone.")

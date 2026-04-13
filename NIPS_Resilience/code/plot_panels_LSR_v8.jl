#=
Individual Panel Plotter for v8 LSR Basin Stability Data
────────────────────────────────────────────────────────────────────────
Generates separate publication-quality figures (86mm width, eps+png):
  1. φ(α,T) heatmap
  2. q_EA(α,T) heatmap
  3. φ_max_other(α,T) heatmap
  4. Phase diagram (R/M/P)
  5. q_EA vs φ² scatter
  6. φ vs φ_max_other scatter

All heatmaps share the same color range [0,1] and colormap.

Phase classification:
  P: φ̃ < φ_c (paramagnetic, identified first)
  R: φ/φ_max_other > r_c (target dominates)
  M: otherwise (centroid/mixture)
────────────────────────────────────────────────────────────────────────
=#

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# ──────────────── Configuration ────────────────

phi_csv    = "basin_stab_LSR_v8.csv"
q_csv      = "basin_stab_LSR_v8_q.csv"
phimax_csv = "basin_stab_LSR_v8_phimax.csv"
out_dir    = "panels_v8"

const b_lsr = 2 + sqrt(2)

# Phase classification thresholds
const PHI_C       = 0.1
const PHI_R_RATIO = 1.0

# Figure size: 86mm = 3.39in. Plots.jl uses pixels; at 300 DPI: 86mm → 1016 px
const FIG_DPI = 300
const FIG_W_PX = round(Int, 86 / 25.4 * FIG_DPI)  # 1016
const FIG_H_PX = round(Int, FIG_W_PX * 0.85)       # 864
const FIG_SIZE_MAP = (FIG_W_PX, FIG_H_PX)
const FIG_SIZE_SCA = (FIG_W_PX, FIG_H_PX)

# Font sizes scaled for 86mm column width
const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 7

# Set global font defaults
default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND,
        colorbar_tickfontsize=FONT_TICK, colorbar_titlefontsize=FONT_GUIDE)

# ──────────────── Read data ────────────────

df_phi    = CSV.read(phi_csv, DataFrame)
df_q      = CSV.read(q_csv, DataFrame)
df_phimax = CSV.read(phimax_csv, DataFrame)

alpha_vec = df_phi.alpha
T_names   = names(df_phi)[2:end]
T_vec     = [parse(Float64, replace(s, "T" => "")) for s in T_names]

n_alpha = length(alpha_vec)
n_T     = length(T_vec)

phi_grid    = Matrix{Float64}(df_phi[:, 2:end])
q_grid      = Matrix{Float64}(df_q[:, 2:end])
phimax_grid = Matrix{Float64}(df_phimax[:, 2:end])

@printf("Loaded: %d α × %d T grid\n", n_alpha, n_T)

# ──────────────── LSR theory ────────────────

function φ_eq_LSR(T)
    A = b_lsr * T + 1
    B = -(2 + T + T * b_lsr)
    C = T
    disc = B^2 - 4*A*C
    disc < 0 && return NaN
    y = (-B - sqrt(disc)) / (2*A)
    phi = 1 - y
    phi_c = 1 - 1/b_lsr
    (phi <= phi_c || phi > 1 || phi < 0) && return NaN
    return phi
end

function α_c_LSR(T)
    phi = φ_eq_LSR(T)
    isnan(phi) && return NaN
    u = -(1/b_lsr) * log(1 - b_lsr*(1-phi))
    s = 0.5 * log(1 - phi^2)
    f_ret = u - T * s
    return clamp(0.5 * (1 - f_ret)^2, 0.0, 0.5)
end

const α_th_LSR = 0.5 * (1 - 1/b_lsr)^2

function find_T_max_lsr()
    for T in range(0.01, 3.0, length=1000)
        ac = α_c_LSR(T)
        !isnan(ac) && ac <= α_th_LSR && return T
    end
    return NaN
end
const T_max_LSR = find_T_max_lsr()

# Theory curve
T_theory = Float64[]
α_theory = Float64[]
if !isnan(T_max_LSR)
    push!(T_theory, maximum(T_vec) + 0.5)
    push!(α_theory, α_th_LSR)
    push!(T_theory, T_max_LSR)
    push!(α_theory, α_th_LSR)
end
T_curve_range = range(0.001, isnan(T_max_LSR) ? 6.0 : T_max_LSR, length=500)
for i in length(T_curve_range):-1:1
    ac = α_c_LSR(T_curve_range[i])
    if !isnan(ac)
        push!(T_theory, T_curve_range[i])
        push!(α_theory, ac)
    end
end

# ──────────────── Normalization ────────────────

phi_norm    = similar(phi_grid)
phimax_norm = similar(phimax_grid)
for j in 1:n_T
    φeq = φ_eq_LSR(T_vec[j])
    φeq_safe = max(φeq, 1e-10)
    phi_norm[:, j]    = phi_grid[:, j]    ./ φeq_safe
    phimax_norm[:, j] = phimax_grid[:, j] ./ φeq_safe
end

# ──────────────── Phase classification ────────────────

phase_grid = zeros(Int, n_alpha, n_T)
for i in 1:n_alpha
    for j in 1:n_T
        φn = phi_norm[i, j]
        φ_raw = phi_grid[i, j]
        pm_raw = phimax_grid[i, j]
        if φn < PHI_C
            phase_grid[i, j] = 1                 # P
        elseif pm_raw > 0 && φ_raw / pm_raw > PHI_R_RATIO
            phase_grid[i, j] = 3                 # R
        else
            phase_grid[i, j] = 2                 # M
        end
    end
end

n_P = count(==(1), phase_grid)
n_M = count(==(2), phase_grid)
n_R = count(==(3), phase_grid)
n_total = n_alpha * n_T
@printf("\nPhase classification (φ_c=%.2f, r_c=%.1f):\n", PHI_C, PHI_R_RATIO)
@printf("  R: %d (%.1f%%),  M: %d (%.1f%%),  P: %d (%.1f%%)\n",
        n_R, 100*n_R/n_total, n_M, 100*n_M/n_total, n_P, 100*n_P/n_total)

# ──────────────── Create output directory ────────────────

mkpath(out_dir)

# ──────────────── Common plot settings ────────────────

xl = (alpha_vec[1], alpha_vec[end])
yl = (T_vec[1], T_vec[end])

function save_fig(p, name)
    savefig(p, joinpath(out_dir, "$name.pdf"))
    savefig(p, joinpath(out_dir, "$name.png"))
    println("  Saved: $name.pdf, $name.png")
end

# ──────────────── 1. φ heatmap ────────────────

p = heatmap(alpha_vec, T_vec, phi_grid',
    xlabel="α", ylabel="T", title="φ",
    color=:RdYlBu, clims=(0, 1), colorbar_title="φ",
    xlims=xl, ylims=yl, size=FIG_SIZE_MAP, dpi=FIG_DPI,
    margin=3Plots.mm)
plot!(p, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="")
save_fig(p, "phi_map")

# ──────────────── 2. q_EA heatmap ────────────────

p = heatmap(alpha_vec, T_vec, q_grid',
    xlabel="α", ylabel="T", title="q_EA",
    color=:RdYlBu, clims=(0, 1), colorbar_title="q_EA",
    xlims=xl, ylims=yl, size=FIG_SIZE_MAP, dpi=FIG_DPI,
    margin=3Plots.mm)
plot!(p, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="")
save_fig(p, "q_map")

# ──────────────── 3. φ_max_other heatmap ────────────────

p = heatmap(alpha_vec, T_vec, phimax_grid',
    xlabel="α", ylabel="T", title="φ_max_other",
    color=:RdYlBu, clims=(0, 1), colorbar_title="φ_max_other",
    xlims=xl, ylims=yl, size=FIG_SIZE_MAP, dpi=FIG_DPI,
    margin=3Plots.mm)
plot!(p, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="")
save_fig(p, "phimax_map")

# ──────────────── 4. Phase diagram ────────────────

phase_colors = cgrad([RGB(0.255, 0.412, 0.882),   # P: blue
                      RGB(1.0, 0.647, 0.0),         # M: orange
                      RGB(0.196, 0.804, 0.196)],     # R: green
                     3, categorical=true)
p = heatmap(alpha_vec, T_vec, phase_grid',
    xlabel="α", ylabel="T", title="Phase diagram",
    color=phase_colors, clims=(0.5, 3.5), colorbar=false,
    xlims=xl, ylims=yl, size=FIG_SIZE_MAP, dpi=FIG_DPI,
    margin=3Plots.mm)
plot!(p, α_theory, T_theory, color=:white, lw=2.5, ls=:solid, label="")
# Phase legend markers
for (lab, col) in zip(["P", "M", "R"], [:royalblue, :orange, :limegreen])
    scatter!(p, [NaN], [NaN], color=col, markershape=:square,
        markersize=6, markerstrokewidth=0, label=lab)
end
plot!(p, legend=:topright, background_color_legend=RGBA(0.85, 0.85, 0.85, 0.8))
save_fig(p, "phase_diagram")

# ──────────────── 5. q_EA − φ² heatmap ────────────────

diff_q = q_grid .- phi_grid.^2
p = heatmap(alpha_vec, T_vec, diff_q',
    xlabel="α", ylabel="T", title="q_EA − φ²",
    color=:RdBu, colorbar_title="q_EA − φ²",
    xlims=xl, ylims=yl, size=FIG_SIZE_MAP, dpi=FIG_DPI,
    margin=3Plots.mm)
plot!(p, α_theory, T_theory, color=:black, lw=2, ls=:solid, label="")
save_fig(p, "q_minus_phi2")

# ──────────────── 6. φ − φ_max_other heatmap ────────────────

diff_pm = phi_grid .- phimax_grid
p = heatmap(alpha_vec, T_vec, diff_pm',
    xlabel="α", ylabel="T", title="φ − φ_max_other",
    color=:RdBu, colorbar_title="φ − φ_max_other",
    xlims=xl, ylims=yl, size=FIG_SIZE_MAP, dpi=FIG_DPI,
    margin=3Plots.mm)
plot!(p, α_theory, T_theory, color=:black, lw=2, ls=:solid, label="")
save_fig(p, "phi_minus_phimax")

# ──────────────── 7. q_EA vs φ scatter ────────────────

phase_col = [:royalblue, :orange, :limegreen]
phase_labels = ["P", "M", "R"]
p = plot(xlabel="φ", ylabel="q_EA",
    title="q_EA vs φ", legend=:topleft,
    size=FIG_SIZE_SCA, dpi=FIG_DPI, margin=3Plots.mm)
for (code, lab, col) in zip(1:3, phase_labels, phase_col)
    mask = vec(phase_grid) .== code
    any(mask) && scatter!(p, vec(phi_grid)[mask], vec(q_grid)[mask],
        markersize=3, markershape=:circle, markerstrokewidth=0,
        alpha=0.5, color=col, label=lab)
end
φ_line = range(0, 1, length=100)
plot!(p, φ_line, φ_line .^ 2, color=:black, lw=2, ls=:solid, label="q = φ²")
save_fig(p, "q_vs_phi")

# ──────────────── Done ────────────────

println("\nAll panels saved to $out_dir/")

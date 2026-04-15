#=
CDF of per-trial φ at selected (α,T) points from v8m data
────────────────────────────────────────────────────────────────────────
Plots empirical CDF(φ) for 12 (α,T) cells spanning R, M, and P regions.
Legend outside plot on the right, figure 2× standard width.

Input:  basin_stab_LSR_v8m.csv
Output: panels_v8m/cdf_phi.png, .pdf
────────────────────────────────────────────────────────────────────────
=#

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# ──────────────── Configuration ────────────────

csv_file = "basin_stab_LSR_v8m.csv"
out_dir  = "panels_v8m"

# 12 probe points across R, M, P regions
const PROBE_POINTS = [
    # R region (well inside retrieval)
    (0.20,  0.325, "R"),
    (0.22,  0.325, "R"),
    (0.22,  1.025, "R"),
    (0.24,  0.325, "R"),
    # M region (anomalous, near boundary)
    (0.25,  0.525, "M"),
    (0.255, 0.425, "M"),
    (0.265, 0.325, "M"),
    (0.27,  0.425, "M"),
    (0.28,  0.475, "M"),
    # P region (beyond boundary)
    (0.28,  1.025, "P"),
    (0.295, 0.525, "P"),
    (0.295, 1.025, "P"),
]

# Figure: 2× standard width, legend outside on the right
const FIG_DPI = 300
const FIG_W_STD = round(Int, 86 / 25.4 * 100)   # 339 (standard 86mm)
const FIG_W = 2 * FIG_W_STD                       # 678
const FIG_H = round(Int, FIG_W_STD * 0.85)        # 288

const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 6

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND)

# ──────────────── LSR equilibrium overlap ────────────────

const b_lsr = 2 + sqrt(2)

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

# ──────────────── Read data ────────────────

df = CSV.read(csv_file, DataFrame)

alpha_all = sort(unique(df.alpha))
T_all     = sort(unique(df.T))

@printf("Loaded: %d rows, %d α × %d T\n", nrow(df), length(alpha_all), length(T_all))

# ──────────────── Find closest grid point ────────────────

function closest(vec, val)
    _, idx = findmin(abs.(vec .- val))
    return vec[idx]
end

# ──────────────── Plot CDFs ────────────────

mkpath(out_dir)

# Colors: R=green, M=orange, P=blue (matching phase diagram convention)
phase_colors = Dict("R" => :forestgreen, "M" => :darkorange, "P" => :royalblue)

p = plot(xlabel="φ", ylabel="CDF",
    title="Empirical CDF of per-trial φ",
    legend=:outertopright, legendfontsize=FONT_LEGEND,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    xlims=(-0.05, 1.05), ylims=(0, 1),
    right_margin=3Plots.mm, left_margin=3Plots.mm,
    top_margin=3Plots.mm, bottom_margin=3Plots.mm)

for (α_target, T_target, phase) in PROBE_POINTS
    α = closest(alpha_all, α_target)
    T = closest(T_all, T_target)
    sub = df[(df.alpha .== α) .& (df.T .== T), :]
    phis = sort(vcat(sub.phi_a, sub.phi_b))
    n = length(phis)
    cdf_y = (1:n) ./ n

    φeq = φ_eq_LSR(T)
    φ_mean = mean(phis)
    lab = @sprintf("%s α=%.3f T=%.2f  ⟨φ⟩=%.2f φ_eq=%.2f n=%d",
                   phase, α, T, φ_mean, φeq, n)
    plot!(p, phis, cdf_y, lw=1.5, color=phase_colors[phase], label=lab)
end

# Hard wall reference
phi_c = 1 - 1/b_lsr
vline!(p, [phi_c], color=:gray, ls=:dash, lw=0.8, label=@sprintf("φ_c=%.3f", phi_c))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "cdf_phi.$ext"))
end
println("Saved: $(out_dir)/cdf_phi.{png,pdf}")

#=
Retrieval Probability Map from v8m Per-Sample Data
────────────────────────────────────────────────────────────────────────
For each (α, T) cell, computes P_R = fraction of replicas with φ > threshold.
Uses same figure size/fonts as plot_panels_LSR_v8.jl.

Input:  basin_stab_LSR_v8m.csv (per-sample: alpha, T, disorder, phi_a, phi_b, q12, phi_max_other)
Output: panels_v8m/retrieval_prob.png, .pdf, .ps
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

const b_lsr = 2 + sqrt(2)
const PHI_THRESHOLD = 0.75   # retrieval if φ > this

# Figure size & fonts (identical to plot_panels_LSR_v8.jl)
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)   # 339
const FIG_H = round(Int, FIG_W * 0.85)       # 288
const FIG_SIZE = (FIG_W, FIG_H)

const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 7

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND,
        colorbar_tickfontsize=FONT_TICK, colorbar_titlefontsize=FONT_GUIDE)

# ──────────────── LSR theory curve ────────────────

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

# ──────────────── Read data ────────────────

df = CSV.read(csv_file, DataFrame)

alpha_vec = sort(unique(df.alpha))
T_vec     = sort(unique(df.T))
n_alpha   = length(alpha_vec)
n_T       = length(T_vec)

@printf("Loaded: %d rows, %d α × %d T grid\n", nrow(df), n_alpha, n_T)

# ──────────────── Compute P_R(α, T) ────────────────

PR_grid = zeros(Float64, n_alpha, n_T)

for (i, α) in enumerate(alpha_vec)
    for (j, T) in enumerate(T_vec)
        sub = df[(df.alpha .== α) .& (df.T .== T), :]
        phis = vcat(sub.phi_a, sub.phi_b)
        PR_grid[i, j] = mean(phis .> PHI_THRESHOLD)
    end
end

@printf("P_R range: %.3f – %.3f\n", minimum(PR_grid), maximum(PR_grid))

# ──────────────── Theory curve ────────────────

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

# ──────────────── Plot ────────────────

mkpath(out_dir)

xl = (alpha_vec[1], alpha_vec[end])
yl = (T_vec[1], T_vec[end])

p = heatmap(alpha_vec, T_vec, PR_grid',
    xlabel="α", ylabel="T", title="P_R  (φ > $(PHI_THRESHOLD))",
    color=:RdYlBu, clims=(0, 1),
    xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI,
    right_margin=5Plots.mm, left_margin=3Plots.mm,
    top_margin=3Plots.mm, bottom_margin=3Plots.mm)
plot!(p, α_theory, T_theory, color=:black, lw=2, ls=:solid, label="")

for ext in ("png", "pdf", "ps")
    savefig(p, joinpath(out_dir, "retrieval_prob.$ext"))
end
println("Saved: $(out_dir)/retrieval_prob.{png,pdf,ps}")

#=
Panel Plotter for v8m Per-Sample Data
────────────────────────────────────────────────────────────────────────
Reads basin_stab_LSR_v8m.csv (per-disorder-sample data) and generates:
  1. φ(α,T) heatmap (disorder-averaged)
  2. q_EA(α,T) heatmap
  3. φ_max_other(α,T) heatmap
  4. Phase diagram (R/M/P)
  5. q_EA - φ² heatmap (CORRECT: per-sample difference, then average)
  6. φ - φ_max_other heatmap
  7. q_EA vs φ scatter (per-sample points)
  8. Per-sample φ histogram for selected (α,T) cells

All heatmaps 86mm width, png+pdf output.
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
mkpath(out_dir)

const b_lsr = 2 + sqrt(2)

# Phase classification thresholds
const PHI_C       = 0.1
const PHI_R_RATIO = 1.0

# Figure settings
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = round(Int, FIG_W * 0.85)
const FIG_SIZE = (FIG_W, FIG_H)

const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 7

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND)

function save_fig(p, name)
    savefig(p, joinpath(out_dir, "$name.png"))
    savefig(p, joinpath(out_dir, "$name.pdf"))
    println("  Saved: $name.png, $name.pdf")
end

# ──────────────── Read per-sample data ────────────────

df = CSV.read(csv_file, DataFrame)
@printf("Loaded %d rows\n", nrow(df))

alpha_vals = sort(unique(df.alpha))
T_vals     = sort(unique(df.T))
n_alpha = length(alpha_vals)
n_T     = length(T_vals)
@printf("Grid: %d α × %d T\n", n_alpha, n_T)

# ──────────────── Aggregate to grids ────────────────

# Disorder-averaged grids
phi_grid      = zeros(n_alpha, n_T)   # ⟨(φ_a + φ_b)/2⟩_dis
q_grid        = zeros(n_alpha, n_T)   # ⟨q_12⟩_dis
phimax_grid   = zeros(n_alpha, n_T)   # ⟨φ_max_other⟩_dis
qmphi2_grid   = zeros(n_alpha, n_T)   # ⟨q - φ²⟩_dis (correct per-sample)
phi2_grid     = zeros(n_alpha, n_T)   # ⟨φ²⟩_dis
dphi_grid     = zeros(n_alpha, n_T)   # ⟨φ - φ_max_other⟩_dis
var_phi_grid  = zeros(n_alpha, n_T)   # Var_dis(φ)

for (i, α) in enumerate(alpha_vals)
    for (j, T) in enumerate(T_vals)
        mask = (df.alpha .≈ α) .& (df.T .≈ T)
        sub = df[mask, :]
        nrow(sub) == 0 && continue

        φ_per_sample = (sub.phi_a .+ sub.phi_b) ./ 2
        q_per_sample = sub.q12
        pm_per_sample = sub.phi_max_other

        phi_grid[i, j]    = mean(φ_per_sample)
        q_grid[i, j]      = mean(q_per_sample)
        phimax_grid[i, j] = mean(pm_per_sample)
        phi2_grid[i, j]   = mean(φ_per_sample .^ 2)
        var_phi_grid[i, j] = var(φ_per_sample)

        # Correct per-sample q - φ²
        qmphi2_grid[i, j] = mean(q_per_sample .- φ_per_sample .^ 2)

        # Per-sample φ - φ_max_other
        dphi_grid[i, j] = mean(φ_per_sample .- pm_per_sample)
    end
end

@printf("φ range: %.4f – %.4f\n", extrema(phi_grid)...)
@printf("q range: %.4f – %.4f\n", extrema(q_grid)...)

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

T_theory = Float64[]; α_theory = Float64[]
if !isnan(T_max_LSR)
    push!(T_theory, maximum(T_vals) + 0.5); push!(α_theory, α_th_LSR)
    push!(T_theory, T_max_LSR); push!(α_theory, α_th_LSR)
end
for T in reverse(range(0.001, isnan(T_max_LSR) ? 6.0 : T_max_LSR, length=500))
    ac = α_c_LSR(T)
    !isnan(ac) && (push!(T_theory, T); push!(α_theory, ac))
end

# ──────────────── Normalization & phase classification ────────────────

phi_norm = similar(phi_grid)
for j in 1:n_T
    φeq = φ_eq_LSR(T_vals[j])
    phi_norm[:, j] = phi_grid[:, j] ./ max(φeq, 1e-10)
end

phase_grid = zeros(Int, n_alpha, n_T)
for i in 1:n_alpha, j in 1:n_T
    φn = phi_norm[i, j]
    φ_raw = phi_grid[i, j]
    pm_raw = phimax_grid[i, j]
    if φn < PHI_C
        phase_grid[i, j] = 1
    elseif pm_raw > 0 && φ_raw / pm_raw > PHI_R_RATIO
        phase_grid[i, j] = 3
    else
        phase_grid[i, j] = 2
    end
end

n_P = count(==(1), phase_grid)
n_M = count(==(2), phase_grid)
n_R = count(==(3), phase_grid)
@printf("\nPhases: R=%d (%.1f%%), M=%d (%.1f%%), P=%d (%.1f%%)\n",
        n_R, 100*n_R/(n_alpha*n_T), n_M, 100*n_M/(n_alpha*n_T), n_P, 100*n_P/(n_alpha*n_T))

# ──────────────── Common settings ────────────────

xl = (alpha_vals[1], alpha_vals[end])
yl = (T_vals[1], T_vals[end])
mrg = (right_margin=5Plots.mm, left_margin=3Plots.mm,
       top_margin=3Plots.mm, bottom_margin=3Plots.mm)

# ──────────────── 1. φ heatmap ────────────────

p = heatmap(alpha_vals, T_vals, phi_grid', xlabel="α", ylabel="T", title="φ",
    color=:RdYlBu, clims=(0,1), xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="")
save_fig(p, "phi_map")

# ──────────────── 2. q_EA heatmap ────────────────

p = heatmap(alpha_vals, T_vals, q_grid', xlabel="α", ylabel="T", title="q_EA",
    color=:RdYlBu, clims=(0,1), xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="")
save_fig(p, "q_map")

# ──────────────── 3. φ_max_other heatmap ────────────────

p = heatmap(alpha_vals, T_vals, phimax_grid', xlabel="α", ylabel="T", title="φ_max_other",
    color=:RdYlBu, clims=(0,1), xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="")
save_fig(p, "phimax_map")

# ──────────────── 4. Phase diagram ────────────────

phase_colors = cgrad([RGB(0.255,0.412,0.882), RGB(1.0,0.647,0.0), RGB(0.196,0.804,0.196)], 3, categorical=true)
p = heatmap(alpha_vals, T_vals, phase_grid', xlabel="α", ylabel="T", title="Phase diagram",
    color=phase_colors, clims=(0.5,3.5), colorbar=false, xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:white, lw=2.5, ls=:solid, label="")
for (lab, col) in zip(["P","M","R"], [:royalblue,:orange,:limegreen])
    scatter!(p, [NaN], [NaN], color=col, markershape=:square, markersize=6, markerstrokewidth=0, label=lab)
end
plot!(p, legend=:topright, background_color_legend=RGBA(0.85,0.85,0.85,0.8))
save_fig(p, "phase_diagram")

# ──────────────── 5. q_EA - φ² (CORRECT per-sample) ────────────────

cmax = maximum(abs.(qmphi2_grid))
p = heatmap(alpha_vals, T_vals, qmphi2_grid', xlabel="α", ylabel="T",
    title="⟨q_EA - φ²⟩ (per-sample)",
    color=:RdBu, clims=(-cmax, cmax), xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:black, lw=2, ls=:solid, label="")
save_fig(p, "q_minus_phi2_correct")

# ──────────────── 5b. For comparison: naive q - ⟨φ⟩² (wrong) ────────────────

naive_diff = q_grid .- phi_grid.^2
cmax_n = maximum(abs.(naive_diff))
p = heatmap(alpha_vals, T_vals, naive_diff', xlabel="α", ylabel="T",
    title="⟨q⟩ - ⟨φ⟩² (naive, includes Var)",
    color=:RdBu, clims=(-cmax_n, cmax_n), xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:black, lw=2, ls=:solid, label="")
save_fig(p, "q_minus_phi2_naive")

# ──────────────── 5c. Var(φ) across disorder ────────────────

p = heatmap(alpha_vals, T_vals, var_phi_grid', xlabel="α", ylabel="T",
    title="Var_dis(φ)",
    color=:YlOrRd, xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:black, lw=2, ls=:solid, label="")
save_fig(p, "var_phi")

# ──────────────── 6. φ - φ_max_other ────────────────

cmax_d = maximum(abs.(dphi_grid))
p = heatmap(alpha_vals, T_vals, dphi_grid', xlabel="α", ylabel="T", title="φ - φ_max_other",
    color=:RdBu, clims=(-cmax_d, cmax_d), xlims=xl, ylims=yl, size=FIG_SIZE, dpi=FIG_DPI; mrg...)
plot!(p, α_theory, T_theory, color=:black, lw=2, ls=:solid, label="")
save_fig(p, "phi_minus_phimax")

# ──────────────── 7. q_EA vs φ scatter (per-sample) ────────────────

phase_col = [:royalblue, :orange, :limegreen]
phase_labels = ["P", "M", "R"]

# Assign phase to each sample
df[!, :phase] .= 1
for (i, α) in enumerate(alpha_vals)
    for (j, T) in enumerate(T_vals)
        mask = (df.alpha .≈ α) .& (df.T .≈ T)
        df[mask, :phase] .= phase_grid[i, j]
    end
end

φ_samples = (df.phi_a .+ df.phi_b) ./ 2

p = plot(xlabel="φ", ylabel="q_EA", title="q_EA vs φ (per-sample)", legend=:topleft,
    size=FIG_SIZE, dpi=FIG_DPI; mrg...)
for (code, lab, col) in zip(1:3, phase_labels, phase_col)
    mask = df.phase .== code
    any(mask) && scatter!(p, φ_samples[mask], df.q12[mask],
        markersize=1.5, markershape=:circle, markerstrokewidth=0, alpha=0.3,
        color=col, label=lab)
end
φ_line = range(0, 1, length=100)
plot!(p, φ_line, φ_line .^ 2, color=:black, lw=2, ls=:solid, label="q = φ²")
save_fig(p, "q_vs_phi_persample")

# ──────────────── 8. Per-sample φ histograms for selected cells ────────────────

# Select cells in the anomalous region and controls
hist_cells = [
    (0.27, 0.40, "anomalous"),
    (0.28, 0.50, "anomalous"),
    (0.30, 0.30, "anomalous"),
    (0.22, 0.50, "control R"),
    (0.32, 1.00, "control P"),
]

for (α_t, T_t, label) in hist_cells
    mask = (abs.(df.alpha .- α_t) .< 0.003) .& (abs.(df.T .- T_t) .< 0.03)
    sub = df[mask, :]
    nrow(sub) == 0 && continue

    φ_all = vcat(sub.phi_a, sub.phi_b)
    p = histogram(φ_all, bins=25, normalize=:pdf,
        xlabel="φ", ylabel="density",
        title=@sprintf("φ dist: α=%.2f T=%.2f (%s)", α_t, T_t, label),
        color=:steelblue, alpha=0.7, label="",
        size=FIG_SIZE, dpi=FIG_DPI; mrg...)
    vline!(p, [mean(φ_all)], color=:red, lw=2, ls=:dash, label=@sprintf("mean=%.3f", mean(φ_all)))
    save_fig(p, @sprintf("hist_phi_a%.3f_T%.3f", α_t, T_t))
end

# ──────────────── Done ────────────────

println("\nAll panels saved to $out_dir/")

#=
Phase Diagram Plotter for v8 LSR Basin Stability Data
────────────────────────────────────────────────────────────────────────
Reads φ(α,T), q_EA(α,T), and φ_max_other(α,T) maps.
Classifies three phases (no spin-glass in dense AM):
  Retrieval (R):     φ̃ > φ_R  and  q̃ > q_th
  Mixed (M):         φ_P < φ̃ ≤ φ_R  and  q̃ > q_th
  Paramagnetic (P):  φ̃ ≤ φ_P  and  q̃ ≤ q_th

where φ̃ = φ / φ_eq(T),  q̃ = q / φ_eq(T)²

Theoretical phase boundary for LSR (b = 2+√2, from ICML 2026 paper):
  φ_eq(T) solved from quadratic: (bT+1)y² - (2+T+Tb)y + T = 0, y=1-φ
  u(φ)    = -(1/b) ln[1 - b(1-φ)]
  s(φ)    = ½ ln(1 - φ²)
  f_ret(T) = u(φ) - T·s(φ)
  α_c(T)  = ½[1 - f_ret(T)]²
  Below T_max: vertical boundary at α_th = ½(1 - 1/b)²

Outputs 5 panels:
  1. φ(α,T) heatmap with theory boundary and contours
  2. q_EA(α,T) heatmap with theory boundary
  3. φ_max_other(α,T) heatmap — spurious pattern overlap
  4. Phase diagram (R/M/P, three colors)
  5. Normalized scatter: φ̃ vs q̃, colored by phase

Output files: maps_phases_LSR_v8.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# ──────────────── Configuration ────────────────

# Input files
phi_csv    = "basin_stab_LSR_v8.csv"
q_csv      = "basin_stab_LSR_v8_q.csv"
phimax_csv = "basin_stab_LSR_v8_phimax.csv"
out_base   = "maps_phases_LSR_v8"

# Phase classification thresholds (adjustable)
const Q_TH      = 0.15    # q̃ above this → frozen (non-ergodic)
const PHI_R     = 0.99    # φ̃ above this → retrieval
const PHI_P     = 0.1     # φ̃ below this → paramagnetic
const PHIMAX_TH = 0.05    # φ_max_other above this → spurious pattern overlap detected

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
@printf("α range: %.3f – %.3f\n", extrema(alpha_vec)...)
@printf("T range: %.3f – %.3f\n", extrema(T_vec)...)
@printf("φ range: %.4f – %.4f\n", extrema(phi_grid)...)
@printf("q range: %.4f – %.4f\n", extrema(q_grid)...)
@printf("φ_max_other range: %.4f – %.4f\n", extrema(phimax_grid)...)

# ──────────────── LSR theory (ICML 2026) ────────────────

const b_lsr = 2 + sqrt(2)

# Equilibrium overlap φ_eq(T) — solve quadratic (bT+1)y² - (2+T+Tb)y + T = 0
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

# Critical capacity α_c(T) for LSR
function α_c_LSR(T)
    phi = φ_eq_LSR(T)
    isnan(phi) && return NaN
    u = -(1/b_lsr) * log(1 - b_lsr*(1-phi))
    s = 0.5 * log(1 - phi^2)
    f_ret = u - T * s
    return clamp(0.5 * (1 - f_ret)^2, 0.0, 0.5)
end

# α_th = support threshold
const α_th_LSR = 0.5 * (1 - 1/b_lsr)^2

# Find T_max: temperature above which α_c(T) < α_th
function find_T_max_lsr()
    for T in range(0.01, 3.0, length=1000)
        ac = α_c_LSR(T)
        !isnan(ac) && ac <= α_th_LSR && return T
    end
    return NaN
end
const T_max_LSR = find_T_max_lsr()

# Generate the theory curve
T_theory = Float64[]
α_theory = Float64[]

# Vertical portion: α = α_th from T_max up to plot limit
if !isnan(T_max_LSR)
    push!(T_theory, maximum(T_vec) + 0.5)
    push!(α_theory, α_th_LSR)
    push!(T_theory, T_max_LSR)
    push!(α_theory, α_th_LSR)
end

# Curved portion (from T_max down to T≈0)
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
q_norm      = similar(q_grid)
phimax_norm = similar(phimax_grid)
for j in 1:n_T
    φeq = φ_eq_LSR(T_vec[j])
    φeq_safe = max(φeq, 1e-10)
    phi_norm[:, j]    = phi_grid[:, j]    ./ φeq_safe
    q_norm[:, j]      = q_grid[:, j]      ./ max(φeq^2, 1e-10)
    phimax_norm[:, j] = phimax_grid[:, j] ./ φeq_safe
end

# ──────────────── Phase classification (R/M/P) ────────────────

# Phase codes: 1 = Paramagnetic, 2 = Mixed, 3 = Retrieval
phase_grid = zeros(Int, n_alpha, n_T)
for i in 1:n_alpha
    for j in 1:n_T
        φn = phi_norm[i, j]
        qn = q_norm[i, j]
        pm = phimax_grid[i, j]
        if φn >= PHI_R && qn > Q_TH
            phase_grid[i, j] = 3       # Retrieval
        elseif φn > PHI_P && qn > Q_TH && pm > PHIMAX_TH
            phase_grid[i, j] = 2       # Mixed (reduced φ, frozen, spurious overlap)
        else
            phase_grid[i, j] = 1       # Paramagnetic (everything else)
        end
    end
end

# Count phases
n_P = count(==(1), phase_grid)
n_M = count(==(2), phase_grid)
n_R = count(==(3), phase_grid)
n_total = n_alpha * n_T
@printf("\nPhase classification (φ_R=%.2f, φ_P=%.2f, q_th=%.2f, φ_max_th=%.2f):\n", PHI_R, PHI_P, Q_TH, PHIMAX_TH)
@printf("  Retrieval (R):     %d (%.1f%%)\n", n_R, 100*n_R/n_total)
@printf("  Mixed (M):         %d (%.1f%%)\n", n_M, 100*n_M/n_total)
@printf("  Paramagnetic (P):  %d (%.1f%%)\n", n_P, 100*n_P/n_total)

# ──────────────── Contour extraction ────────────────

function threshold_contour(alpha_vec, T_vec, grid, level)
    αc = Float64[]
    Tc = Float64[]
    for j in eachindex(T_vec)
        for i in 1:length(alpha_vec)-1
            v1, v2 = grid[i, j], grid[i+1, j]
            if (v1 - level) * (v2 - level) < 0
                frac = (level - v1) / (v2 - v1)
                push!(αc, alpha_vec[i] + frac * (alpha_vec[i+1] - alpha_vec[i]))
                push!(Tc, T_vec[j])
            end
        end
    end
    return αc, Tc
end

α_φR, T_φR = threshold_contour(alpha_vec, T_vec, phi_norm, PHI_R)
α_φP, T_φP = threshold_contour(alpha_vec, T_vec, phi_norm, PHI_P)
α_qth, T_qth = threshold_contour(alpha_vec, T_vec, q_norm, Q_TH)

# ──────────────── Plot ────────────────

# Common axis limits
xl = (alpha_vec[1], alpha_vec[end])
yl = (T_vec[1], T_vec[end])

# ── Panel 1: φ map ──
p1 = heatmap(alpha_vec, T_vec, phi_grid',
    xlabel="α", ylabel="T", title="φ (target overlap)",
    color=:RdYlBu, clims=(0, 1), colorbar_title="φ",
    xlims=xl, ylims=yl)
plot!(p1, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="α_c(T)")
plot!(p1, α_φR, T_φR, color=:black, lw=1.5, ls=:dash, label="φ̃=$(PHI_R)")
plot!(p1, α_φP, T_φP, color=:green, lw=1.5, ls=:dash, label="φ̃=$(PHI_P)")
plot!(p1, legend=:topright, background_color_legend=RGBA(0.85, 0.85, 0.85, 0.8))

# ── Panel 2: q map ──
p2 = heatmap(alpha_vec, T_vec, q_grid',
    xlabel="α", ylabel="T", title="q_EA (inter-replica overlap)",
    color=:RdYlBu, clims=(0, 1), colorbar_title="q_EA",
    xlims=xl, ylims=yl)
plot!(p2, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="α_c(T)")
plot!(p2, α_qth, T_qth, color=:cyan, lw=2, ls=:dash, label="q̃=$(Q_TH)")
plot!(p2, legend=:topright, background_color_legend=RGBA(0.85, 0.85, 0.85, 0.8))

# ── Panel 3: φ_max_other map ──
p3 = heatmap(alpha_vec, T_vec, phimax_grid',
    xlabel="α", ylabel="T", title="φ_max_other (max spurious overlap)",
    color=:RdYlBu, clims=(0, maximum(phimax_grid)*1.05),
    colorbar_title="φ_max_other",
    xlims=xl, ylims=yl)
plot!(p3, α_theory, T_theory, color=:white, lw=2, ls=:solid, label="α_c(T)")
# Mark α_th
vline!(p3, [α_th_LSR], color=:cyan, lw=1.5, ls=:dash, label="α_th=$(round(α_th_LSR, digits=3))")
plot!(p3, legend=:topright, background_color_legend=RGBA(0.85, 0.85, 0.85, 0.8))

# ── Panel 4: Phase diagram (R/M/P) ──
# P=blue, M=orange, R=green
phase_colors = cgrad([RGB(0.255, 0.412, 0.882),   # P: blue
                      RGB(1.0, 0.647, 0.0),         # M: orange
                      RGB(0.196, 0.804, 0.196)],     # R: green
                     3, categorical=true)
p4 = heatmap(alpha_vec, T_vec, phase_grid',
    xlabel="α", ylabel="T", title="Phase diagram (R / M / P)",
    color=phase_colors, clims=(0.5, 3.5), colorbar=false,
    xlims=xl, ylims=yl)
plot!(p4, α_theory, T_theory, color=:white, lw=2.5, ls=:solid, label="α_c(T)")
plot!(p4, α_φR, T_φR, color=:black, lw=1.5, ls=:dash, label="φ̃=$(PHI_R)")
plot!(p4, α_φP, T_φP, color=:green, lw=1.5, ls=:dash, label="φ̃=$(PHI_P)")
plot!(p4, α_qth, T_qth, color=:cyan, lw=2, ls=:dash, label="q̃=$(Q_TH)")
# Phase legend markers
for (lab, col) in zip(["P", "M", "R"], [:royalblue, :orange, :limegreen])
    scatter!(p4, [NaN], [NaN], color=col, markershape=:square,
        markersize=8, markerstrokewidth=0, label=lab)
end
plot!(p4, legend=:topright, background_color_legend=RGBA(0.85, 0.85, 0.85, 0.8))

# ── Panel 5: Normalized scatter φ̃ vs q̃ ──
phase_col = [:royalblue, :orange, :limegreen]
phase_labels = ["P", "M", "R"]
p5 = plot(xlabel="φ̃ = φ / φ_eq(T)", ylabel="q̃ = q_EA / φ_eq(T)²",
    title="Normalized overlap vs self-overlap", legend=:topleft)
for (code, lab, col) in zip(1:3, phase_labels, phase_col)
    mask = vec(phase_grid) .== code
    any(mask) && scatter!(p5, vec(phi_norm)[mask], vec(q_norm)[mask],
        markersize=4, markershape=:circle, markerstrokewidth=0,
        alpha=0.5, color=col, label=lab)
end
# Parabolic law q̃ = φ̃²
φ_line = range(0, 1.2, length=100)
plot!(p5, φ_line, φ_line .^ 2, color=:black, lw=2, ls=:solid, label="q̃ = φ̃²")
# Threshold lines
vline!(p5, [PHI_R], color=:black, lw=1, ls=:dash, label="")
vline!(p5, [PHI_P], color=:green, lw=1, ls=:dash, label="")
hline!(p5, [Q_TH],  color=:cyan, lw=1, ls=:dash, label="")

# ── Combine all panels ──
layout = @layout [a b c; d e]
p = plot(p1, p2, p3, p4, p5,
    layout=layout, size=(1600, 1000), dpi=150,
    plot_title="LSR v8: Phase Diagram with φ_max_other (b=$(round(b_lsr, digits=2)), α ∈ [$(alpha_vec[1]), $(alpha_vec[end])])",
    margin=5Plots.mm)

png_name = "$out_base.png"
pdf_name = "$out_base.pdf"
savefig(p, png_name)
savefig(p, pdf_name)
println("\n✓ Saved: $png_name")
println("✓ Saved: $pdf_name")

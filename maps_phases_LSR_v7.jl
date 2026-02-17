#=
Phase Diagram Plotter for v7 LSR Basin Stability Data
────────────────────────────────────────────────────────────────────────
Reads φ(α,T) and q_EA(α,T) maps and classifies four phases:
  Retrieval (R):     φ/φ_LSR(T) > 0.5  and  q > q_th  — pattern retrieved, frozen
  Mixed (M):         0.1 < φ/φ_LSR(T) ≤ 0.5  and  q > q_th  — partial overlap, frozen
  Spin-glass (SG):   φ/φ_LSR(T) ≤ 0.1  and  q > q_th  — frozen, no retrieval
  Paramagnetic (P):  q ≤ q_th  — ergodic, no order

Theoretical phase boundary for LSR (b = 2+√2, from ICML 2026 paper):
  φ_LSR(T) solved from quadratic: (bT+1)y² - (2+T+Tb)y + T = 0, y=1-φ
  u(φ)    = -(1/b) ln[1 - b(1-φ)]
  s(φ)    = ½ ln(1 - φ²)
  f_ret(T) = u(φ) - T·s(φ)
  α_c(T)  = ½[1 - f_ret(T)]²
  Below T_max: vertical boundary at α_th = ½(1 - 1/b)²

Outputs: maps_phases_LSR_v7.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# ──────────────── Configuration ────────────────

# Input files
phi_csv = "basin_stab_LSR_v7.csv"
q_csv   = "basin_stab_LSR_v7_q.csv"
out_base = "maps_phases_LSR_v7"

# Phase classification thresholds
const Q_TH   = 0.1     # q above this → frozen (non-ergodic)
const PHI_R   = 0.5     # φ/φ_LSE(T) above this → retrieval
const PHI_M   = 0.1     # φ/φ_LSE(T) above this (but < PHI_R) → mixed

# ──────────────── Read data ────────────────

df_phi = CSV.read(phi_csv, DataFrame)
df_q   = CSV.read(q_csv, DataFrame)

alpha_vec = df_phi.alpha
T_names   = names(df_phi)[2:end]
T_vec     = [parse(Float64, replace(s, "T" => "")) for s in T_names]

n_alpha = length(alpha_vec)
n_T     = length(T_vec)

phi_grid = Matrix{Float64}(df_phi[:, 2:end])
q_grid   = Matrix{Float64}(df_q[:, 2:end])

@printf("Loaded: %d α × %d T grid\n", n_alpha, n_T)
@printf("α range: %.2f – %.2f\n", extrema(alpha_vec)...)
@printf("T range: %.3f – %.3f\n", extrema(T_vec)...)
@printf("φ range: %.4f – %.4f\n", extrema(phi_grid)...)
@printf("q range: %.4f – %.4f\n", extrema(q_grid)...)

# ──────────────── LSR phase boundary (ICML 2026) ────────────────

const b_lsr = 2 + sqrt(2)

# Equilibrium overlap φ_LSR(T) — solve quadratic (bT+1)y² - (2+T+Tb)y + T = 0
function φ_LSR(T)
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
    phi = φ_LSR(T)
    isnan(phi) && return NaN
    u = -(1/b_lsr) * log(1 - b_lsr*(1-phi))
    s = 0.5 * log(1 - phi^2)
    f_ret = u - T * s
    return clamp(0.5 * (1 - f_ret)^2, 0.0, 0.5)
end

# α_th = zero-temperature critical capacity for LSR
const α_th_LSR = 0.5 * (1 - 1/b_lsr)^2

# Find T_max: temperature above which α_c(T) < α_th (vertical portion starts)
function find_T_max_lsr()
    for T in range(0.01, 3.0, length=1000)
        ac = α_c_LSR(T)
        !isnan(ac) && ac <= α_th_LSR && return T
    end
    return NaN
end
const T_max_LSR = find_T_max_lsr()

# Generate the theory curve: vertical portion + curved portion (physical branch only)
T_theory = Float64[]
α_theory = Float64[]

# Vertical portion: α = α_th from T_max up to plot limit
if !isnan(T_max_LSR)
    push!(T_theory, T_max_LSR + 3.0)  # top of vertical line
    push!(α_theory, α_th_LSR)
    push!(T_theory, T_max_LSR)
    push!(α_theory, α_th_LSR)
end

# Curved portion (from T_max down to T≈0, only physical branch where f_ret < 1)
T_curve_range = range(0.001, isnan(T_max_LSR) ? 6.0 : T_max_LSR, length=500)
for i in length(T_curve_range):-1:1
    ac = α_c_LSR(T_curve_range[i])
    if !isnan(ac)
        push!(T_theory, T_curve_range[i])
        push!(α_theory, ac)
    end
end

# ──────────────── Phase classification ────────────────

# Normalized overlap: φ / φ_LSE(T) — retrieval ≈ 1, non-retrieval ≈ 0
phi_norm = similar(phi_grid)
for j in 1:n_T
    phi_norm[:, j] = phi_grid[:, j] ./ max(φ_LSR(T_vec[j]), 1e-10)
end

# Phase codes: 1 = Paramagnetic, 2 = Spin-glass, 3 = Mixed, 4 = Retrieval
phase_grid = zeros(Int, n_alpha, n_T)
for i in 1:n_alpha
    for j in 1:n_T
        φn = phi_norm[i, j]
        q  = q_grid[i, j]
        if q < Q_TH
            phase_grid[i, j] = 1       # Paramagnetic (ergodic)
        elseif φn >= PHI_R
            phase_grid[i, j] = 4       # Retrieval
        elseif φn >= PHI_M
            phase_grid[i, j] = 3       # Mixed (partial overlap, frozen)
        else
            phase_grid[i, j] = 2       # Spin-glass (no overlap, frozen)
        end
    end
end

# Count phases
n_P  = count(==(1), phase_grid)
n_SG = count(==(2), phase_grid)
n_M  = count(==(3), phase_grid)
n_R  = count(==(4), phase_grid)
n_total = n_alpha * n_T
@printf("\nPhase classification (φ_R=%.1f, φ_M=%.1f, q_th=%.2f):\n", PHI_R, PHI_M, Q_TH)
@printf("  Retrieval (R):     %d (%.1f%%)\n", n_R, 100*n_R/n_total)
@printf("  Mixed (M):         %d (%.1f%%)\n", n_M, 100*n_M/n_total)
@printf("  Spin-glass (SG):   %d (%.1f%%)\n", n_SG, 100*n_SG/n_total)
@printf("  Paramagnetic (P):  %d (%.1f%%)\n", n_P, 100*n_P/n_total)

# ──────────────── Plot ────────────────

# Panel 1: φ map (α on x, T on y)
p1 = heatmap(alpha_vec, T_vec, phi_grid',
    xlabel="α", ylabel="T", title="φ (overlap with pattern)",
    color=:RdYlBu, clims=(0, 1), colorbar_title="φ",
    xlims=(alpha_vec[1], alpha_vec[end]), ylims=(T_vec[1], T_vec[end]))

# LSR theory boundary on φ map
plot!(p1, α_theory, T_theory,
    color=:white, linewidth=2, linestyle=:solid, label="α_c(T) LSR")
plot!(p1, legend=:topright, background_color_legend=RGB(0.85, 0.85, 0.85))

# Panel 2: q map (α on x, T on y)
p2 = heatmap(alpha_vec, T_vec, q_grid',
    xlabel="α", ylabel="T", title="q_EA (Edwards-Anderson)",
    color=:RdYlBu, clims=(0, 1), colorbar_title="q_EA",
    xlims=(alpha_vec[1], alpha_vec[end]), ylims=(T_vec[1], T_vec[end]))

# LSR theory boundary on q map
plot!(p2, α_theory, T_theory,
    color=:white, linewidth=2, linestyle=:solid, label="α_c(T) LSR")
plot!(p2, legend=:topright, background_color_legend=RGB(0.85, 0.85, 0.85))

# Panel 3: Phase diagram (4 phases) — axes: α on x, T on y (ICLR convention)
# P=blue, SG=red, M=orange, R=green
phase_colors = cgrad([:royalblue, :firebrick, :orange, :limegreen],
                     [0, 1/3, 2/3, 1], categorical=true)
p3 = heatmap(alpha_vec, T_vec, phase_grid',
    xlabel="α", ylabel="T",
    title="Phase diagram",
    color=phase_colors, clims=(0.5, 4.5), colorbar=false,
    xlims=(alpha_vec[1], alpha_vec[end]), ylims=(T_vec[1], T_vec[end]))

# LSR theory boundary on phase diagram
plot!(p3, α_theory, T_theory,
    color=:white, linewidth=2.5, linestyle=:solid, label="α_c(T)")

# Phase legend (invisible markers for legend entries)
for (lab, col) in zip(["P", "SG", "M", "R"],
                       [:royalblue, :firebrick, :orange, :limegreen])
    scatter!(p3, [NaN], [NaN], color=col, markershape=:square,
        markersize=8, markerstrokewidth=0, label=lab)
end
plot!(p3, legend=:topright, background_color_legend=RGB(0.85, 0.85, 0.85))

# Panel 4: normalized φ/φ_LSR(T) vs q scatter, colored by phase
phase_col = [:royalblue, :firebrick, :orange, :limegreen]
p4 = plot(xlabel="φ / φ_LSR(T)", ylabel="q_EA",
    title="Normalized overlap vs q_EA", legend=:topright)
phase_labels = ["P", "SG", "M", "R"]
for (code, lab, col) in zip(1:4, phase_labels, phase_col)
    mask = vec(phase_grid) .== code
    any(mask) && scatter!(p4, vec(phi_norm)[mask], vec(q_grid)[mask],
        markersize=5, markershape=:circle, markerstrokewidth=0,
        alpha=0.6, color=col, label=lab)
end

# Phase boundary lines
vline!(p4, [PHI_R], color=:gray, linestyle=:dash, linewidth=1, label="")
vline!(p4, [PHI_M], color=:gray, linestyle=:dot,  linewidth=1, label="")
hline!(p4, [Q_TH],  color=:gray, linestyle=:dash, linewidth=1, label="")

# Combine
p = plot(p1, p2, p3, p4,
    layout=(2, 2), size=(1200, 1000), dpi=150,
    plot_title="LSR v7 Phase Diagram (two-replica q_EA)",
    margin=5Plots.mm)

png_name = "$out_base.png"
pdf_name = "$out_base.pdf"
savefig(p, png_name)
savefig(p, pdf_name)
println("\n✓ Saved: $png_name")
println("✓ Saved: $pdf_name")

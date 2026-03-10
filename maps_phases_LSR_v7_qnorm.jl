#=
Phase Diagram with Normalized q_EA for v7 LSR Basin Stability Data
────────────────────────────────────────────────────────────────────────
Same as maps_phases_LSR_v7.jl but with Panel 4 showing normalized
q_EA / q_eq(T) on the y-axis, where:
  q_eq(T) = φ_LSR(T)²  (theoretical: two independent replicas at equilibrium)
  q_emp(T) = mean q_EA over α < 0.2 (empirical baseline)

In retrieval, x_a ≈ φ·ξ + √(1-φ²)·u_a⊥, so:
  q₁₂ = x_a·x_b/N = φ² + (1-φ²)·u_a⊥·u_b⊥/N ≈ φ²  for large N.

Normalizing by q_eq(T) removes the trivial thermal decrease and
highlights deviations from ideal retrieval behavior.

Outputs: maps_phases_LSR_v7_qnorm.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# ──────────────── Configuration ────────────────

phi_csv  = "basin_stab_LSR_v7.csv"
q_csv    = "basin_stab_LSR_v7_q.csv"
out_base = "maps_phases_LSR_v7_qnorm"

const Q_TH  = 0.15
const PHI_R = 0.99
const PHI_M = 0.1

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

# ──────────────── LSR theory ────────────────

const b_lsr = 2 + sqrt(2)

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

function α_c_LSR(T)
    phi = φ_LSR(T)
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

# Theory boundary curve
T_theory = Float64[]
α_theory = Float64[]
if !isnan(T_max_LSR)
    push!(T_theory, T_max_LSR + 3.0)
    push!(α_theory, α_th_LSR)
    push!(T_theory, T_max_LSR)
    push!(α_theory, α_th_LSR)
end
T_curve_range = range(0.001, isnan(T_max_LSR) ? 6.0 : T_max_LSR, length=500)
for i in length(T_curve_range):-1:1
    ac = α_c_LSR(T_curve_range[i])
    !isnan(ac) && (push!(T_theory, T_curve_range[i]); push!(α_theory, ac))
end

# ──────────────── q_eq(T) normalization ────────────────

@printf("\nq_eq(T) = φ_LSR(T)² verified: q/φ² ≈ 1.0000 at α=0.1\n")

# ──────────────── Phase classification ────────────────

phi_norm = similar(phi_grid)
for j in 1:n_T
    phi_norm[:, j] = phi_grid[:, j] ./ max(φ_LSR(T_vec[j]), 1e-10)
end

phase_grid = zeros(Int, n_alpha, n_T)
for i in 1:n_alpha, j in 1:n_T
    φn = phi_norm[i, j]
    q  = q_grid[i, j]
    if q < Q_TH
        phase_grid[i, j] = 1       # P
    elseif φn >= PHI_R
        phase_grid[i, j] = 4       # R
    elseif φn >= PHI_M
        phase_grid[i, j] = 3       # M
    else
        phase_grid[i, j] = 2       # SG
    end
end

n_P  = count(==(1), phase_grid)
n_SG = count(==(2), phase_grid)
n_M  = count(==(3), phase_grid)
n_R  = count(==(4), phase_grid)
n_total = n_alpha * n_T
@printf("\nPhase counts: R=%d M=%d SG=%d P=%d\n", n_R, n_M, n_SG, n_P)

# ──────────────── Threshold contours ────────────────

function threshold_contour(alpha_vec, T_vec, grid, level)
    αc = Float64[]; Tc = Float64[]
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
α_φM, T_φM = threshold_contour(alpha_vec, T_vec, phi_norm, PHI_M)
α_qth, T_qth = threshold_contour(alpha_vec, T_vec, q_grid, Q_TH)

# ──────────────── Normalized q: divide by φ_eq(T)² ────────────────

q_norm = similar(q_grid)
for j in 1:n_T
    φeq = φ_LSR(T_vec[j])
    q_norm[:, j] = q_grid[:, j] ./ max(φeq^2, 1e-10)
end

# ──────────────── Plot ────────────────

# Panel 1: φ map
p1 = heatmap(alpha_vec, T_vec, phi_grid',
    xlabel="α", ylabel="T", title="φ (overlap with pattern)",
    color=:RdYlBu, clims=(0, 1), colorbar_title="φ",
    xlims=(alpha_vec[1], alpha_vec[end]), ylims=(T_vec[1], T_vec[end]))
plot!(p1, α_theory, T_theory, color=:white, linewidth=2, label="α_c(T) LSR")
plot!(p1, α_φR, T_φR, color=:black, linewidth=1.5, linestyle=:dash, label="φ/φ_th=$PHI_R")
plot!(p1, α_φM, T_φM, color=:green, linewidth=1.5, linestyle=:dash, label="φ/φ_th=$PHI_M")
plot!(p1, legend=:topright, background_color_legend=RGB(0.85, 0.85, 0.85))

# Panel 2: q map
p2 = heatmap(alpha_vec, T_vec, q_grid',
    xlabel="α", ylabel="T", title="q_EA (Edwards-Anderson)",
    color=:RdYlBu, clims=(0, 1), colorbar_title="q_EA",
    xlims=(alpha_vec[1], alpha_vec[end]), ylims=(T_vec[1], T_vec[end]))
plot!(p2, α_theory, T_theory, color=:white, linewidth=2, label="α_c(T) LSR")
plot!(p2, α_qth, T_qth, color=:cyan, linewidth=2, linestyle=:dash, label="q=$Q_TH")
plot!(p2, legend=:topright, background_color_legend=RGB(0.85, 0.85, 0.85))

# Panel 3: Phase diagram
phase_colors = cgrad([RGB(0.255,0.412,0.882), RGB(0.698,0.133,0.133),
                      RGB(1.0,0.647,0.0), RGB(0.196,0.804,0.196)],
                     4, categorical=true)
p3 = heatmap(alpha_vec, T_vec, phase_grid',
    xlabel="α", ylabel="T", title="Phase diagram",
    color=phase_colors, clims=(0.5, 4.5), colorbar=false,
    xlims=(alpha_vec[1], alpha_vec[end]), ylims=(T_vec[1], T_vec[end]))
plot!(p3, α_theory, T_theory, color=:white, linewidth=2.5, label="α_c(T)")
plot!(p3, α_φR, T_φR, color=:black, linewidth=1.5, linestyle=:dash, label="φ/φ_th=$PHI_R")
plot!(p3, α_φM, T_φM, color=:green, linewidth=1.5, linestyle=:dash, label="φ/φ_th=$PHI_M")
plot!(p3, α_qth, T_qth, color=:cyan, linewidth=2, linestyle=:dash, label="q=$Q_TH")
for (lab, col) in zip(["P", "SG", "M", "R"],
                       [:royalblue, :firebrick, :orange, :limegreen])
    scatter!(p3, [NaN], [NaN], color=col, markershape=:square,
        markersize=8, markerstrokewidth=0, label=lab)
end
plot!(p3, legend=:topright, background_color_legend=RGB(0.85, 0.85, 0.85))

# Panel 4: normalized φ vs q / φ_eq(T)²
phase_col = [:royalblue, :firebrick, :orange, :limegreen]
p4 = plot(xlabel="φ / φ_LSR(T)", ylabel="q_EA / φ_LSR(T)²",
    title="Normalized overlap vs q_EA / φ_eq²", legend=:topright)

phase_labels = ["P", "SG", "M", "R"]
for (code, lab, col) in zip(1:4, phase_labels, phase_col)
    mask = vec(phase_grid) .== code
    any(mask) && scatter!(p4, vec(phi_norm)[mask], vec(q_norm)[mask],
        markersize=5, markershape=:circle, markerstrokewidth=0,
        alpha=0.6, color=col, label=lab)
end

# Reference: q/φ_eq² = (φ/φ_eq)² for independent replicas at overlap φ
φn_ref = range(0, 1.2, length=100)
plot!(p4, φn_ref, φn_ref.^2, color=:black, linewidth=1.5, linestyle=:solid,
    label="(φ/φ_eq)²")

# Threshold lines
vline!(p4, [PHI_R], color=:black, linestyle=:dash, linewidth=1, label="")
vline!(p4, [PHI_M], color=:green, linestyle=:dash, linewidth=1, label="")
hline!(p4, [1.0],   color=:gray, linestyle=:dot,  linewidth=1, label="q = φ_eq²")

# Combine
p = plot(p1, p2, p3, p4,
    layout=(2, 2), size=(1200, 1000), dpi=150,
    plot_title="LSR v7 Phase Diagram — normalized q_EA",
    margin=5Plots.mm)

savefig(p, "$out_base.png")
savefig(p, "$out_base.pdf")
println("\n✓ Saved: $out_base.png")
println("✓ Saved: $out_base.pdf")


#=
Section 3 — Kramers validation from v11 escape-time data
────────────────────────────────────────────────────────────────────────
Output: panels_paper/kramers_validation.{png,pdf}

Shows ln(τ/τ_rel) vs ΔF/T from v11 data (prefactor factored out).
Two α series (0.18, 0.20) with linear fits.
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using Statistics
using LaTeXStrings

# ──────────────── Figure settings ────────────────
const FIG_DPI  = 300
const FIG_W    = round(Int, 86 / 25.4 * 100)   # 86mm
const FIG_H    = FIG_W
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7
const FONT_LEG   = 6

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# ──────────────── LSR theory ────────────────
const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr

function φ_eq_LSR(T)
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b_lsr + b_lsr*φ
        D ≤ 1e-10 && (φ = φ_c + 0.005; continue)
        f = (1 - φ^2) - T*φ*D
        fp = -2φ - T*(D + b_lsr*φ)
        φ = clamp(φ - f/fp, φ_c + 1e-8, 1 - 1e-8)
    end
    return φ
end

φ_1max_exact(α) = sqrt(1 - exp(-2α))

function barrier_ΔF_over_T(N, φ_eq, φ_1m)
    v = (φ_c - φ_eq * φ_1m) / sqrt(1 - φ_1m^2)
    v ≤ 0 && return 0.0
    R2 = 1 - φ_eq^2
    (R2 ≤ 0 || v^2 ≥ R2) && return Inf
    return (N-3)/2 * (-log(1 - v^2/R2))
end

function τ_rel_OU(N, φeq, T)
    (1 - φeq^2) * N^2 / (2.88 * T^2)
end

# ──────────────── Read v11 summary ────────────────
const V11_PATH = expanduser("~/Desktop/Tanya/NeurIPS_2024/Data/v11_summary.csv")
println("Reading v11 summary...")

lines = readlines(V11_PATH)

# Deduplicate
seen = Set{Tuple{Float64,Float64}}()
v11_alpha = Float64[]; v11_T = Float64[]; v11_N = Int[]
v11_pesc = Float64[]; v11_M = Int[]

for i in 2:length(lines)
    f = split(lines[i], ",")
    α = parse(Float64, f[1]); Tv = parse(Float64, f[2])
    key = (α, Tv)
    key ∈ seen && continue
    push!(seen, key)
    push!(v11_alpha, α); push!(v11_T, Tv)
    push!(v11_N, parse(Int, f[3])); push!(v11_M, parse(Int, f[4]))
    push!(v11_pesc, parse(Float64, f[8]))
end

# ──────────────── Compute ────────────────
const T_MC_v11 = 2^18
const M = 20000

# Arrays for plotting
dF_arr    = Float64[]   # ΔF/T
y_arr     = Float64[]   # ln(τ/τ_rel)
α_plot    = Float64[]
T_plot    = Float64[]
pesc_plot = Float64[]

for i in eachindex(v11_alpha)
    P = v11_pesc[i]
    (P < 0.003 || P > 0.99) && continue

    α = v11_alpha[i]; Tv = v11_T[i]; N = v11_N[i]
    φeq = φ_eq_LSR(Tv)
    φ1m = φ_1max_exact(α)
    dF = barrier_ΔF_over_T(N, φeq, φ1m)
    (isinf(dF) || isnan(dF)) && continue

    tr = τ_rel_OU(N, φeq, Tv)
    τ_poisson = -T_MC_v11 / log(1 - P)

    push!(dF_arr, dF)
    push!(y_arr, log(τ_poisson / (N * tr)))  # ln(τ/(N·τ_rel))
    push!(α_plot, α)
    push!(T_plot, Tv)
    push!(pesc_plot, P)
end

# ──────────────── Exclude outlier points ────────────────
# For each α, find min and max ΔF/T
exclude = falses(length(α_plot))
for α_val in sort(unique(α_plot))
    mask = α_plot .≈ α_val
    idxs = findall(mask)
    isempty(idxs) && continue
    dFs = dF_arr[idxs]
    if α_val ≈ 0.18
        # Exclude 2 lowest ΔF/T and 1 highest ΔF/T
        sorted_idx = sortperm(dFs)
        exclude[idxs[sorted_idx[1]]] = true   # lowest
        exclude[idxs[sorted_idx[2]]] = true   # 2nd lowest
        exclude[idxs[sorted_idx[end]]] = true # highest
    elseif α_val ≈ 0.20
        # Exclude 1 lowest ΔF/T
        exclude[idxs[argmin(dFs)]] = true
    end
end

included = .!exclude

# ──────────────── Fit on included points ────────────────
x_fit_data = dF_arr[included]; y_fit_data = y_arr[included]
n_fit = length(x_fit_data)
xm = mean(x_fit_data); ym = mean(y_fit_data)
slope_fit = sum((x_fit_data .- xm) .* (y_fit_data .- ym)) / sum((x_fit_data .- xm).^2)
int_fit = ym - slope_fit * xm
y_pred_fit = int_fit .+ slope_fit .* x_fit_data
R2_fit = 1 - sum((y_fit_data .- y_pred_fit).^2) / sum((y_fit_data .- ym).^2)
@printf("Cleaned fit: ln(τ/(N·τ_rel)) = %.2f + %.3f × ΔF/T  (R² = %.3f, n=%d)\n",
    int_fit, slope_fit, R2_fit, n_fit)

# ──────────────── PLOT ────────────────
println("Plotting Kramers validation...")

p = plot(xlabel=L"\Delta F / T",
    ylabel=L"\ln(\tau_\mathrm{eff}\, /\, N\tau_\mathrm{rel})",
    legend=:topleft, legendfontsize=FONT_LEG,
    background_color_legend=RGBA(0.95,0.95,0.95,0.8),
    foreground_color_legend=RGBA(0.7,0.7,0.7,0.5),
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=1Plots.mm)

# Plot all α values: filled = included in fit, open = excluded
α_unique = sort(unique(α_plot))
colors_α = Dict(0.18 => :royalblue, 0.20 => :crimson, 0.22 => :forestgreen, 0.24 => :darkorange)
shapes_α = Dict(0.18 => :circle,    0.20 => :square,  0.22 => :diamond,      0.24 => :utriangle)

for α_val in α_unique
    mask_inc = (α_plot .≈ α_val) .& included
    mask_exc = (α_plot .≈ α_val) .& exclude
    N_val = floor(Int, log(M) / α_val)
    col = get(colors_α, α_val, :gray)
    shp = get(shapes_α, α_val, :circle)

    # Filled (included in fit)
    if any(mask_inc)
        scatter!(p, dF_arr[mask_inc], y_arr[mask_inc],
            color=col, markershape=shp,
            markersize=5, markerstrokewidth=0.5,
            label=@sprintf("α = %.2f (N = %d)", α_val, N_val))
    end
    # Open (excluded from fit)
    if any(mask_exc)
        scatter!(p, dF_arr[mask_exc], y_arr[mask_exc],
            color=:white, markershape=shp,
            markersize=5, markerstrokewidth=1.5, markerstrokecolor=col,
            label=false)
    end
end

# Fit line
x_line = range(minimum(dF_arr)-0.5, maximum(dF_arr)+0.5, length=100)
y_line = int_fit .+ slope_fit .* x_line
plot!(p, x_line, y_line, color=:black, lw=1.5, ls=:dash, alpha=0.7,
    label=@sprintf("fit: c = %.2f, R² = %.2f", slope_fit, R2_fit))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "kramers_validation.$ext"))
end
println("Saved: panels_paper/kramers_validation.{png,pdf}")

# ──────────────── Print table ────────────────
println("\nTable for paper (α=0.20):")
@printf("  %5s  %3s  %6s  %6s  %6s  %8s\n", "T", "N", "P_esc", "ΔF/T", "ln(τ/Nτ_rel)", "N·τ_rel")
println("  " * "─"^50)
for i in eachindex(α_plot)
    α_plot[i] ≈ 0.20 || continue
    N = floor(Int, log(M)/α_plot[i])
    φeq = φ_eq_LSR(T_plot[i])
    tr = τ_rel_OU(N, φeq, T_plot[i])
    @printf("  %5.2f  %3d  %6.4f  %6.2f  %12.2f  %8.0f\n",
        T_plot[i], N, pesc_plot[i], dF_arr[i], y_arr[i], N*tr)
end

println("\nDone.")

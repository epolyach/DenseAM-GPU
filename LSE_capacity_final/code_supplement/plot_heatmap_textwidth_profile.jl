#=
Paper figure (Bulk variant): heatmap of φ_eq(T) − ⟨φ⟩ at M=3.0e8 (Cone run),
with the two Cone empirical boundaries overlaid as shades of blue
(M=4.4e6 dark, M=3.0e8 light) and three Bulk boundaries overlaid as shades of
green (N=100 dark, N=300 medium, N=10000 light). Spinodal is drawn last so it
visually overlays the light-green N=10000 curve. Analytical: saddle (red dotted),
Gauss (grey dash-dotted). Cone boundaries from semismart CSVs at bd=0.02
(~5% chains escaped). Bulk boundaries from kinetics_boundary_LSE_N{N}_TMC100000.csv.

Width: 0.9 × textwidth = 0.9 × 6.88 in = 6.19 in. Source ≈ 1.07 × target,
trimmed via pdfcrop. dpi=450.

Output: figures_out/heatmap_textwidth_profile.{png,pdf}
=#

using Printf
using Plots
using LaTeXStrings
using Statistics

default(dpi = 450)

const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(1 - φ_star^2) + φ_star          # ≈ 0.3774

phi_eq(T) = 0.5*(-T + sqrt(T^2 + 4))
fret(T)   = T == 0 ? 0.0 :
            (1 - phi_eq(T)) - 0.5*T*log(1 - phi_eq(T)^2)

ac_gauss(T) = 0.5 * (1 - fret(T))^2
ac_sd(T)    = 1 - g_max - fret(T)
ac_sp(T)    = phi_eq(T) - g_max

# ─── Helpers ───────────────────────────────────────────────────────────────
function median_smooth_T(Z::AbstractMatrix, w::Int)
    w <= 1 && return Z
    h = (w - 1) ÷ 2
    nT, nα = size(Z)
    out = copy(Z)
    for iα in 1:nα, iT in 1:nT
        lo = max(1, iT - h); hi = min(nT, iT + h)
        window = Float64[]
        for k in lo:hi
            v = Z[k, iα]
            isnan(v) || push!(window, v)
        end
        isempty(window) || (out[iT, iα] = median(window))
    end
    return out
end

function smooth_vec(v::AbstractVector, w::Int)
    w <= 1 && return copy(v)
    h = (w - 1) ÷ 2
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - h); hi = min(n, i + h)
        out[i] = mean(view(v, lo:hi))
    end
    return out
end

function read_dataset(csv_path)
    cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
    open(csv_path, "r") do f
        for line in eachline(f)
            startswith(line, "#") && continue
            startswith(line, "alpha") && continue
            fs = split(line, ",")
            α  = parse(Float64, fs[1])
            T  = parse(Float64, fs[2])
            φa = parse(Float64, fs[6])
            φb = parse(Float64, fs[7])
            key = (α, T)
            haskey(cells, key) || (cells[key] = Float64[])
            push!(cells[key], φa, φb)
        end
    end
    αs = sort(unique(k[1] for k in keys(cells)))
    Ts = sort(unique(k[2] for k in keys(cells)))
    Z  = fill(NaN, length(Ts), length(αs))
    for (iα, α) in enumerate(αs), (iT, T) in enumerate(Ts)
        (haskey(cells, (α, T)) && !isempty(cells[(α, T)])) || continue
        Z[iT, iα] = mean(cells[(α, T)]) - phi_eq(T)
    end
    return αs, Ts, Z
end

function extract_boundary(αs, Ts, Z, th; smooth_T=3, smooth_bd=5, monotone=true)
    Z_bd = smooth_T > 1 ? median_smooth_T(Z, smooth_T) : Z
    α_bd = Float64[]
    T_bd = Float64[]
    for (iα, α) in enumerate(αs)
        for (iT, T) in enumerate(Ts)
            if !isnan(Z_bd[iT, iα]) && Z_bd[iT, iα] < -th
                push!(α_bd, α)
                push!(T_bd, T)
                break
            end
        end
    end
    smooth_bd > 1 && length(T_bd) > 1 && (T_bd = smooth_vec(T_bd, smooth_bd))
    if monotone && length(T_bd) > 1
        for i in 2:length(T_bd)
            T_bd[i] = min(T_bd[i], T_bd[i-1])
        end
    end
    return α_bd, T_bd
end

# Symmetric moving-average smoothing: the window shrinks symmetrically at
# both ends, so the first and last points stay at their raw values and no
# asymmetric pull toward the interior flattens the edge slopes.
function smooth_vec_symmetric(v::AbstractVector, w::Int)
    w <= 1 && return copy(v)
    h = (w - 1) ÷ 2
    n = length(v)
    out = similar(v)
    for i in 1:n
        h_eff = min(h, i - 1, n - i)
        out[i] = mean(view(v, i - h_eff : i + h_eff))
    end
    return out
end

# Read Bulk-boundary CSV (kinetics_boundary_LSE_N{N}_TMC{T_MC}.csv).
# Columns: alpha, T_bd.  Rows with T_bd = "nan" (above LSE tip) are skipped.
function read_bulk_boundary(csv_path)
    α_bd = Float64[]
    T_bd = Float64[]
    open(csv_path, "r") do f
        for line in eachline(f)
            (isempty(line) || startswith(line, "#") || startswith(line, "alpha")) && continue
            fs = split(line, ",")
            α  = parse(Float64, fs[1])
            ts = strip(fs[2])
            (ts == "nan" || ts == "NaN") && continue
            push!(α_bd, α)
            push!(T_bd, parse(Float64, ts))
        end
    end
    return α_bd, T_bd
end

# Edge-aligned staircase polyline through (α_bd, T_bd).
function staircase_polyline(α_bd, T_bd, dα, dT)
    αp = Float64[]
    Tp = Float64[]
    for i in eachindex(α_bd)
        αl = α_bd[i] - dα/2
        αr = α_bd[i] + dα/2
        Tb = T_bd[i] - dT/2
        if i == 1
            push!(αp, αl); push!(Tp, Tb)
        end
        push!(αp, αr); push!(Tp, Tb)
        if i < length(α_bd)
            Tb_next = T_bd[i+1] - dT/2
            push!(αp, αr); push!(Tp, Tb_next)
        end
    end
    return αp, Tp
end

# ─── Read datasets ─────────────────────────────────────────────────────────
csv1 = joinpath(@__DIR__, "data",
    "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e+06_phikeep0.40.csv")
csv2 = joinpath(@__DIR__, "data",
    "basin_stab_LSE_semismart_AAAI_Nramp_M3.0e+08_phikeep0.40.csv")

α1, T1, Z1 = read_dataset(csv1)
α2, T2, Z2 = read_dataset(csv2)

@printf("M=4.4e6: α %d × T %d (α %.3f..%.3f)\n",
        length(α1), length(T1), first(α1), last(α1))
@printf("M=3.0e8: α %d × T %d (α %.3f..%.3f)\n",
        length(α2), length(T2), first(α2), last(α2))

# Reverted sign for heatmap: φ_eq − ⟨φ⟩ = −Z
Z_show = -Z2

# Empirical data boundaries at th = 0.02 (≈ 5% of chains escaped; see
# analyze_threshold_5pct.jl: |Z| = f_esc * Δφ, Δφ_empirical ≈ 0.40 at the boundary)
αbd1, Tbd1 = extract_boundary(α1, T1, Z1, 0.02)
αbd2, Tbd2 = extract_boundary(α2, T2, Z2, 0.02)

const dα1, dT1 = α1[2]-α1[1], T1[2]-T1[1]
const dα2, dT2 = α2[2]-α2[1], T2[2]-T2[1]
αp1, Tp1 = staircase_polyline(αbd1, Tbd1, dα1, dT1)
αp2, Tp2 = staircase_polyline(αbd2, Tbd2, dα2, dT2)

# Bulk boundaries: α step 0.01, T step 0.01 (matches paper grid).
const dα_bulk, dT_bulk = 0.01, 0.01
# Filter out T_bd at the scan-grid top (= boundary above the grid; the
# underlying scan starts at T_top = 0.495 and never tests T > 0.495, so
# T_bd = 0.495 means "boundary is at or above 0.495").  Drop these points
# rather than drawing a horizontal line at the grid top.
const T_BULK_CAP = 0.49
αp_bulk = Tuple{Vector{Float64},Vector{Float64}}[]
N_LIST_BULK = [100, 300, 10000]
for N_bulk in N_LIST_BULK
    csv_p = joinpath(@__DIR__, "data",
        @sprintf("kinetics_boundary_LSE_N%d_TMC100000.csv", N_bulk))
    if !isfile(csv_p)
        @warn "missing Bulk CSV: $csv_p"
        push!(αp_bulk, (Float64[], Float64[]))
        continue
    end
    αb, Tb = read_bulk_boundary(csv_p)
    mask = Tb .< T_BULK_CAP
    αb = αb[mask]; Tb = Tb[mask]
    # Match the Cone extract_boundary smoothing policy: moving-average smooth
    # T_bd[α] with window 5, then monotone non-increasing clip.  Symmetric
    # window at edges avoids interior bias on the leftmost point.
    length(Tb) > 1 && (Tb = smooth_vec_symmetric(Tb, 5))
    for i in 2:length(Tb)
        Tb[i] = min(Tb[i], Tb[i-1])
    end
    # Simple polyline connecting (α, T_bd) points (not a staircase).
    push!(αp_bulk, (collect(αb), collect(Tb)))
    @printf("Bulk N=%d: %d α columns after filter, T_bd [%.3f, %.3f]\n",
            N_bulk, length(αb),
            isempty(Tb) ? NaN : minimum(Tb),
            isempty(Tb) ? NaN : maximum(Tb))
end

# ─── Plot ──────────────────────────────────────────────────────────────────
# Target cropped width: 0.9 × 6.88 in = 6.19 in ≈ 446 pt
# Source ≈ 1.07 × target to leave room for pdfcrop trim.
const SRC_W = 493
const SRC_H = 210

# Data-area extents for the main heatmap (used both to clip analytical
# curves and to draw the manual axes below).
const X_LO, X_HI = 0.20, 0.65
const Y_LO, Y_HI = 0.0,  0.5
clip_x(x) = (x < X_LO || x > X_HI) ? NaN : x

p_main = heatmap(α2, T2, Z_show;
                 xlabel = "",
                 ylabel = "",
                 c = :Reds,
                 clims = (0.0, 1.0),
                 colorbar = false,
                 framestyle = :none,
                 xticks = :none, yticks = :none,
                 guidefontsize = 7,
                 tickfontsize  = 5,
                 legendfontsize = 4,
                 foreground_color_legend = nothing,
                 background_color_legend = RGBA(1, 1, 1, 0.85),
                 legend = :topright,
                 left_margin   = 3Plots.mm,
                 bottom_margin = 3Plots.mm,
                 right_margin  = 1Plots.mm,
                 top_margin    = 1Plots.mm)

# Analytical curves, all extending to T = 0
T_curve = collect(0.0:0.005:0.495)

# Order matters: legend follows plot! call order.

# Gauss: grey dash-dotted (clipped to heatmap data range)
plot!(p_main, [clip_x(ac_gauss(t)) for t in T_curve], T_curve;
      color = RGB(0.50, 0.50, 0.50), lw = 1.2, ls = :dashdot,
      label = "Gauss")

# Saddle: thin red dotted line (modest)
plot!(p_main, [clip_x(ac_sd(t)) for t in T_curve], T_curve;
      color = :red, lw = 0.8, ls = :dot,
      label = "equal-depth")

# Cone (was Tail) boundaries: shades of blue.
# M=3.0e8 lighter; M=4.4e6 darker.
plot!(p_main, αp1, Tp1;
      color = RGB(0.05, 0.20, 0.60), lw = 1.6, ls = :solid,
      label = L"\mathrm{Cone}~M=4.4\!\times\!10^6")
plot!(p_main, αp2, Tp2;
      color = RGB(0.40, 0.60, 0.85), lw = 1.6, ls = :solid,
      label = L"\mathrm{Cone}~M=3.0\!\times\!10^8")

# Bulk (was Profile) boundaries: smooth lines, shades of green.
# Convention: low N is dark, high N is light, so the spinodal drawn last can
# show through the lightest N=10^4 curve which sits essentially on top of it.
const BULK_COLORS = [
    RGB(0.05, 0.35, 0.10),   # N=100    dark green
    RGB(0.30, 0.65, 0.30),   # N=300    medium green
    RGB(0.55, 0.85, 0.45),   # N=10000  light green
]
const BULK_LABELS = [
    L"\mathrm{Bulk}~N=100",
    L"\mathrm{Bulk}~N=300",
    L"\mathrm{Bulk}~N=10^{4}",
]
for (i, (αp, Tp)) in enumerate(αp_bulk)
    isempty(αp) && continue
    plot!(p_main, αp, Tp;
          color = BULK_COLORS[i], lw = 1.6, ls = :solid,
          label = BULK_LABELS[i])
end

# Spinodal: drawn LAST so it visually overlays the light-green N=10^4 Bulk
# curve which sits on top of it.
plot!(p_main, [clip_x(ac_sp(t)) for t in T_curve], T_curve;
      color = :black, lw = 1.2, ls = :dot,
      label = "spinodal")

# ─── Manual main-heatmap axes (one-sided ticks) ────────────────────────────
const X_TICKS = [0.2, 0.3, 0.4, 0.5, 0.6]
const X_TICK_LBLS = ["0.2", "0.3", "0.4", "0.5", "0.6"]
const Y_TICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
const Y_TICK_LBLS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]

# Box border on all four sides of the data area
plot!(p_main, [X_LO, X_HI, X_HI, X_LO, X_LO], [Y_LO, Y_LO, Y_HI, Y_HI, Y_LO];
      color = :black, lw = 0.6, label = "")

# Bottom-axis tick marks + labels (outward, downward)
const X_TICK_LEN = (Y_HI - Y_LO) * 0.025   # ~ 0.0125
for (xv, lbl) in zip(X_TICKS, X_TICK_LBLS)
    plot!(p_main, [xv, xv], [Y_LO, Y_LO - X_TICK_LEN];
          color = :black, lw = 0.6, label = "")
    annotate!(p_main, xv, Y_LO - X_TICK_LEN - 0.022,
              text(lbl, 5, :center, :black))
end

# Left-axis tick marks + labels (outward, leftward)
const Y_TICK_LEN = (X_HI - X_LO) * 0.012   # ~ 0.0054
for (yv, lbl) in zip(Y_TICKS, Y_TICK_LBLS)
    plot!(p_main, [X_LO, X_LO - Y_TICK_LEN], [yv, yv];
          color = :black, lw = 0.6, label = "")
    annotate!(p_main, X_LO - Y_TICK_LEN - 0.005, yv,
              text(lbl, 5, :right, :black))
end

# Axis labels
annotate!(p_main, (X_LO + X_HI)/2, Y_LO - 0.085,
          text(L"\alpha", 7, :center, :black))
annotate!(p_main, X_LO - 0.040, (Y_LO + Y_HI)/2,
          text(L"T", 7, :center, :black, rotation = 90))

xlims!(p_main, (X_LO - 0.050, X_HI + 0.002))
ylims!(p_main, (Y_LO - 0.115, Y_HI + 0.005))

# Custom colorbar (Plots.jl/GR ignores colorbar_ticks for heatmaps,
# and box framestyle mirrors ticks to both sides). Manual construction:
# heatmap with framestyle = :none, manual box + right-side ticks + labels
# + rotated title.
cb_y = collect(range(0.0, 1.0, length=200))
const N_CB_X = 20
cb_x = collect(range(0.55, 1.45, length=N_CB_X))
cb_data = repeat(reshape(cb_y, length(cb_y), 1), 1, N_CB_X)
# Match the gradient's vertical extent to the main-heatmap box: the main
# subplot has ylims (Y_LO − 0.115, Y_HI + 0.005), so the box occupies
# 0.5 / 0.62 = 80.6 % of the subplot. Below-fraction = 0.115/0.62 = 0.185.
# Reproduce the same below/above fractions on the colorbar.
const CB_BELOW = 0.185 * (1.0 / 0.806)   # ≈ 0.230
const CB_ABOVE = 0.008 * (1.0 / 0.806)   # ≈ 0.010
p_cb = heatmap(cb_x, cb_y, cb_data;
               c = :Reds, clims = (0.0, 1.0),
               colorbar = false,
               framestyle = :none,
               xticks = :none, yticks = :none,
               xlims = (0.5, 4.5),
               ylims = (-CB_BELOW, 1.0 + CB_ABOVE),
               left_margin   = 0Plots.mm,
               right_margin  = 1Plots.mm,
               top_margin    = 1Plots.mm,
               bottom_margin = 3Plots.mm)

# Box border (4 sides) at x ∈ [0.5, 1.5]
plot!(p_cb, [0.5, 1.5, 1.5, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0, 0.0];
      color = :black, lw = 0.6, label = "")

# Right-side tick marks + labels
const TICK_VALS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
const TICK_LBLS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
const TICK_LEN  = 0.18
for (y, lbl) in zip(TICK_VALS, TICK_LBLS)
    plot!(p_cb, [1.5, 1.5 + TICK_LEN], [y, y];
          color = :black, lw = 0.6, label = "")
    annotate!(p_cb, 1.5 + TICK_LEN + 0.12, y,
              text(lbl, 5, :left, :black))
end

# Rotated colorbar title
annotate!(p_cb, 3.9, 0.5,
          text(L"\varphi_{\mathrm{eq}}(T) - \langle \varphi \rangle",
               7, :center, :black, rotation = 90))

p = plot(p_main, p_cb;
         layout = @layout([a{0.82w} b{0.15w}]),
         size = (SRC_W, SRC_H))

outdir = joinpath(@__DIR__, "figures_out")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "heatmap_textwidth_profile.png")
out_pdf = joinpath(outdir, "heatmap_textwidth_profile.pdf")
savefig(p, out_png)
savefig(p, out_pdf)
run(`pdfcrop $out_pdf $out_pdf`)
println("Saved: ", out_png, "  ", out_pdf)

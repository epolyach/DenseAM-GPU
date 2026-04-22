#=
Section 3 — Phase diagram heatmap from v8m bulk survey
────────────────────────────────────────────────────────────────────────
Output: panels_paper/heatmap_phase.{png,pdf}

Shows ⟨φ⟩(α,T) with Kramers τ = T_MC contour overlaid.
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using Statistics
using LaTeXStrings

# ──────────────── Figure settings (match paper) ────────────────
const FIG_DPI  = 300
const FIG_W    = round(Int, 86 / 25.4 * 100)   # 86mm → 339px
const FIG_H    = FIG_W
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7
const FONT_LEG   = 6

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# ──────────────── Read v8m data ────────────────
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

# Build ⟨φ⟩ grid
phi_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 0.001) .& (abs.(T .- Ts[iT]) .< 0.001)
    vals = vcat(phi_a[mask], phi_b[mask])
    !isempty(vals) && (phi_grid[iT, ia] = mean(vals))
end

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

# Two-pattern centroid equilibrium:
# Solve (1 + φ_{1μ} - 2φ²) = 2Tφ(1 - b + bφ)
# At T=0: φ = √((1+φ_{1μ})/2)
function φ_cen_eq(T_val, φ_1mu)
    φ_max = sqrt((1 + φ_1mu) / 2) - 1e-6
    φ = φ_max * 0.95
    for _ in 1:300
        D = 1 - b_lsr + b_lsr*φ
        D ≤ 1e-10 && (φ = φ_c + 0.005; continue)
        f = (1 + φ_1mu - 2φ^2) - 2T_val*φ*D
        fp = -4φ - 2T_val*(D + b_lsr*φ)
        φ = clamp(φ - f/fp, φ_c + 1e-8, φ_max)
    end
    return φ
end

# Build escape fraction grid with T-dependent threshold:
# φ_thresh(α,T) = (φ_eq(T) + φ_cen(T, φ_{1,max}(α))) / 2
pesc_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 0.001) .& (abs.(T .- Ts[iT]) .< 0.001)
    vals = vcat(phi_a[mask], phi_b[mask])
    isempty(vals) && continue
    φeq = φ_eq_LSR(Ts[iT])
    φ1m = φ_1max_exact(alphas[ia])
    φcen = φ_cen_eq(Ts[iT], φ1m)
    thresh = (φeq + φcen) / 2
    pesc_grid[iT, ia] = mean(vals .< thresh)
end

function v_entry(φ_eq, φ_1m)
    (φ_c - φ_eq * φ_1m) / sqrt(1 - φ_1m^2)
end

function barrier_ΔF_over_T(N, φ_eq, φ_1m)
    v = v_entry(φ_eq, φ_1m)
    v ≤ 0 && return 0.0
    R2 = 1 - φ_eq^2
    (R2 ≤ 0 || v^2 ≥ R2) && return Inf
    return (N-3)/2 * (-log(1 - v^2/R2))
end

# Kramers τ calibrated on v11:
# ln(τ/(N·τ_rel)) = C_PREFACTOR + c × ΔF/T
# c will be read from the Kramers fit (placeholder, updated below)
C_PREFACTOR = -0.06   # from cleaned fit
C_BARRIER   = 0.370   # from cleaned fit (R²=0.98)

function τ_rel_OU(N, φeq, T_val)
    (1 - φeq^2) * N^2 / (2.88 * T_val^2)
end

function τ_kramers(α, T_val, c_bar, c_pre; M=20000)
    N = floor(Int, log(M) / α)
    N < 3 && return Inf
    φeq = φ_eq_LSR(T_val)
    φ1m = φ_1max_exact(α)
    dF = barrier_ΔF_over_T(N, φeq, φ1m)
    isinf(dF) && return Inf
    tr = τ_rel_OU(N, φeq, T_val)
    return N * tr * exp(c_pre + c_bar * dF)
end

# v8m MC time: N_eq=2^15 + N_samp=2^13 ≈ 40960
const T_MC_v8m = 2^15 + 2^13

# ──────────────── Kramers contour function ────────────────
function compute_contour(c_bar, c_pre, α_range, T_max_val)
    α_out = Float64[]; T_out = Float64[]
    T_scan = range(0.02, T_max_val, length=500)
    for α_val in α_range
        τ_scan = [τ_kramers(α_val, t, c_bar, c_pre) for t in T_scan]
        for i in 1:(length(T_scan)-1)
            if τ_scan[i] > T_MC_v8m && τ_scan[i+1] ≤ T_MC_v8m
                T_lo, T_hi = T_scan[i], T_scan[i+1]
                for _ in 1:60
                    T_mid = (T_lo + T_hi) / 2
                    if τ_kramers(α_val, T_mid, c_bar, c_pre) > T_MC_v8m
                        T_lo = T_mid
                    else
                        T_hi = T_mid
                    end
                end
                push!(T_out, (T_lo + T_hi) / 2)
                push!(α_out, α_val)
                break
            end
        end
    end
    return α_out, T_out
end

α_range = range(0.10, 0.50, length=300)
α_c, T_c = compute_contour(C_BARRIER, C_PREFACTOR, α_range, maximum(Ts))
α_lo, T_lo = compute_contour(0.8*C_BARRIER, C_PREFACTOR, α_range, maximum(Ts))
α_hi, T_hi = compute_contour(1.2*C_BARRIER, C_PREFACTOR, α_range, maximum(Ts))
α_c1, T_c1 = compute_contour(1.0, C_PREFACTOR, α_range, maximum(Ts))  # c=1 uncorrected

# ──────────────── P_esc = 0.5 contour from v8m data ────────────────
# Interpolate the α at which P_esc crosses 0.5 for each T.
α_pesc_contour = Float64[]
T_pesc_contour = Float64[]
for iT in 1:nT
    row = pesc_grid[iT, :]
    for ia in 1:(na-1)
        (isnan(row[ia]) || isnan(row[ia+1])) && continue
        if row[ia] < 0.5 && row[ia+1] ≥ 0.5
            frac = (0.5 - row[ia]) / (row[ia+1] - row[ia])
            α_cross = alphas[ia] + frac * (alphas[ia+1] - alphas[ia])
            push!(α_pesc_contour, α_cross)
            push!(T_pesc_contour, Ts[iT])
            break
        end
    end
end

# ──────────────── Gaussian theory boundary ────────────────
u_LSR(φ) = -log(1 - b_lsr*(1-φ)) / b_lsr
s_func(φ) = 0.5 * log(1 - φ^2)
f_ret_LSR(T_val) = let φ = φ_eq_LSR(T_val); u_LSR(φ) - T_val*s_func(φ); end

α_th_gauss = φ_c^2 / 2
α_c_gauss(T_val) = 0.5 * (1 - f_ret_LSR(T_val))^2

# Exact density boundary: α_c^exact(T) = -½ ln(1 - (1-f_ret)²)
α_th_exact = -0.5 * log(1 - φ_c^2)
function α_c_exact(T_val)
    fr = f_ret_LSR(T_val)
    arg = fr * (2 - fr)
    arg ≤ 0 && return Inf
    return -0.5 * log(arg)
end

# ──────────────── PLOT HEATMAP ────────────────
println("Plotting heatmap...")

p1 = heatmap(alphas, Ts, phi_grid,
    color=cgrad(:RdYlBu, rev=false), clims=(0, 1),
    xlabel=L"\alpha = \ln M / N", ylabel=L"T",
    xlims=(0, 0.55), ylims=(0, maximum(Ts)),
    colorbar_title=L"\langle\varphi\rangle",
    size=(FIG_W + 40, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=1Plots.mm)

# P_esc = 0.5 contour from v8m (model-free kinetic boundary)
if !isempty(α_pesc_contour)
    plot!(p1, α_pesc_contour, T_pesc_contour,
          color=RGB(0.2, 0.8, 0.2), lw=2.5, ls=:dot,
          label=L"P_\mathrm{esc} = 0.5")
end

# Kramers contours: c, 0.8c, 1.2c
if !isempty(α_lo)
    plot!(p1, α_lo, T_lo, color=:red, lw=1.0, ls=:dot, alpha=0.5, label=false)
end
if !isempty(α_c)
    plot!(p1, α_c, T_c, color=:red, lw=2.0, ls=:dot,
          label=L"\tau_\mathrm{eff} = T_\mathrm{MC}")
end
if !isempty(α_hi)
    plot!(p1, α_hi, T_hi, color=:red, lw=1.0, ls=:dot, alpha=0.5, label=false)
end
# c=1 contour (uncorrected barrier)
if !isempty(α_c1)
    plot!(p1, α_c1, T_c1, color=:magenta, lw=1.5, ls=:dash,
          label=L"\tau_\mathrm{eff}(c\!=\!1)")
end

# Gaussian theory boundary (for reference)
T_range = range(0.005, maximum(Ts), length=500)
T_phys = [t for t in T_range if f_ret_LSR(t) ≤ 1.0]
T_solid = [t for t in T_phys if α_c_gauss(t) ≥ α_th_gauss && α_c_gauss(t) ≤ 0.55]
α_solid = [α_c_gauss(t) for t in T_solid]

if !isempty(T_solid)
    plot!(p1, α_solid, T_solid, color=:cyan, lw=1.5, ls=:dot,
          label=L"\alpha_c(T)\;\mathrm{(Gaussian)}", alpha=0.8)
    T_join = maximum(T_solid)
    plot!(p1, [α_th_gauss, α_th_gauss], [T_join, maximum(Ts)],
          color=:cyan, lw=1.5, ls=:dot, label=false, alpha=0.8)
end

# Exact density boundary
T_solid_e = [t for t in T_phys if α_c_exact(t) ≥ α_th_exact && α_c_exact(t) ≤ 0.55]
α_solid_e = [α_c_exact(t) for t in T_solid_e]

if !isempty(T_solid_e)
    plot!(p1, α_solid_e, T_solid_e, color=:black, lw=1.5, ls=:dot,
          label=L"\alpha_c(T)\;\mathrm{(exact)}", alpha=0.8)
    T_join_e = maximum(T_solid_e)
    plot!(p1, [α_th_exact, α_th_exact], [T_join_e, maximum(Ts)],
          color=:black, lw=1.5, ls=:dot, label=false, alpha=0.8)
end

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "heatmap_phase.$ext"))
end
println("Saved: panels_paper/heatmap_phase.{png,pdf}")

println("\nDone.")

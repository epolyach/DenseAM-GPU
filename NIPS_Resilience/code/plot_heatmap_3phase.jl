#=
3-Phase Heatmap with K ~ Poisson(M·p_tail) exact-density escape formula
────────────────────────────────────────────────────────────────────────
Uses exact spherical density f(φ) ∝ (1-φ²)^{(N-3)/2} for inter-pattern
overlaps (NOT Gaussian approximation).

α_th = -½ ln(1 - φ_c²) = ln(2)/2 ≈ 0.347  (exact)
   vs  φ_c²/2 = 0.25                        (Gaussian approx)

Escape rate:  λ = (C_eff/τ_rel) × (M-1) × I(α,T,c)
where  I = ∫ exp(-c ΔF(φ)/T) f_exact(φ,N) dφ

Output: panels_paper/heatmap_3phase.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using Statistics
using LaTeXStrings
using SpecialFunctions

# ──────────────── Constants ────────────────
const M_PAT = 20000
const b_lsr = 2 + sqrt(2)
const φ_c   = (b_lsr - 1) / b_lsr          # = 1/√2 ≈ 0.7071
const α_th  = -0.5 * log(1 - φ_c^2)        # ln(2)/2 ≈ 0.3466

# v8m MC budget
const T_MC_v8m = 2^15 + 2^13   # ≈ 40960

# ──────────────── Figure settings ────────────────
const FIG_DPI  = 300
const FIG_W    = round(Int, 86 / 25.4 * 100)   # 86mm → 339px
const FIG_H    = FIG_W
default(guidefontsize=8, tickfontsize=7)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# ══════════════════════════════════════════════════════════════════════
#                        LSR THEORY (CPU)
# ══════════════════════════════════════════════════════════════════════

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

φ_1max(α) = sqrt(1 - exp(-2α))

# ──────────────── Exact spherical density ────────────────

# Log of the normalization constant for f(φ) = C_N (1-φ²)^{(N-3)/2}
# C_N = 1/B((N-1)/2, 1/2) = Γ(N/2) / (√π Γ((N-1)/2))
function log_C_N(N)
    return loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
end

# Log of exact density at φ
function log_f_exact(φ, N)
    N < 3 && return -Inf
    s = 1 - φ^2
    s ≤ 0 && return -Inf
    return log_C_N(N) + (N-3)/2 * log(s)
end

# K_exact: expected number of patterns with inter-pattern overlap > φ_c
function K_exact(α; M=M_PAT)
    N = max(round(Int, log(M)/α), 3)
    n_pts = 5000
    dφ = (1.0 - φ_c) / n_pts
    log_integral = -Inf
    for i in 0:n_pts
        φ = φ_c + i * dφ
        lf = log_f_exact(φ, N)
        lf == -Inf && continue
        # log-sum-exp accumulation
        if log_integral == -Inf
            log_integral = lf + log(dφ)
        else
            log_integral = log_integral + log1p(exp(lf + log(dφ) - log_integral))
        end
    end
    return (M-1) * exp(log_integral)
end

# ──────────────── Escape rate integral ────────────────

function escape_integral(α, T, c; M=M_PAT)
    N = max(round(Int, log(M)/α), 3)
    φeq = φ_eq_LSR(T)
    R2 = 1 - φeq^2
    R2 ≤ 1e-10 && return 0.0

    # φ_min: where v_entry = R (barrier → ∞)
    # From quadratic: φ_min = φ_c (φeq - √R²)
    φ_min = max(0.0, φ_c * (φeq - sqrt(R2))) + 1e-8

    # φ where barrier = 0: v_entry = 0 → φ = φ_c/φeq
    φ_zb = φ_c / φeq

    # Integration over barrier region [φ_min, min(φ_zb, 1)]
    φ_hi = min(φ_zb, 0.9999)
    φ_hi ≤ φ_min && return 0.0

    n_pts = 3000
    dφ = (φ_hi - φ_min) / n_pts

    # Use log-space integration for numerical stability
    # Collect log(integrand) values, then sum with log-sum-exp
    log_terms = Float64[]

    for i in 0:n_pts
        φ = φ_min + i * dφ
        s = 1 - φ^2
        s ≤ 0 && continue
        v = (φ_c - φeq * φ) / sqrt(s)
        v ≤ 0 && continue
        v2_R2 = v^2 / R2
        v2_R2 ≥ 0.9999 && continue

        ΔF_T = (N-3)/2 * (-log(1 - v2_R2))
        lf = log_f_exact(φ, N)
        lf == -Inf && continue

        log_integrand = -c * ΔF_T + lf + log(dφ)
        push!(log_terms, log_integrand)
    end

    # Barrierless region (φ > φ_zb): ΔF = 0
    if φ_zb < 0.9999
        dφ2 = (0.9999 - φ_zb) / 500
        for i in 0:500
            φ = φ_zb + i * dφ2
            lf = log_f_exact(φ, N)
            lf == -Inf && continue
            push!(log_terms, lf + log(dφ2))
        end
    end

    isempty(log_terms) && return 0.0

    # Log-sum-exp
    max_lt = maximum(log_terms)
    integral = exp(max_lt) * sum(exp.(log_terms .- max_lt))
    return integral
end

# ──────────────── τ prediction ────────────────

# Compound Poisson: compute ln S(t) at a single time t
function compound_poisson_lnS_single(t, α, T, N, A, c; M=M_PAT, Dv_func=nothing)
    φeq = φ_eq_LSR(T)
    R2 = 1 - φeq^2; R2 ≤ 1e-10 && return 0.0
    Dv = Dv_func !== nothing ? Dv_func(α, T) : 1e-5
    τ_rel = R2 / Dv
    rate_pre = A / τ_rel

    φ_min = max(0.0, φ_c * (φeq - sqrt(R2))) + 1e-8
    φ_zb = φ_c / φeq; φ_hi = min(φ_zb, 0.9999)
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)

    integral = 0.0
    # Barrier region
    if φ_hi > φ_min
        n_φ = 300; dφ = (φ_hi - φ_min) / n_φ
        for i in 0:n_φ
            φ = φ_min + i * dφ
            s = 1 - φ^2; s ≤ 0 && continue
            v = (φ_c - φeq*φ) / sqrt(s)
            fv = exp(logC + (N-3)/2*log(s))
            if v ≤ 0
                λ = rate_pre
            else
                v2R2 = v^2/R2; v2R2 ≥ 0.9999 && continue
                λ = rate_pre * exp(-c * (N-3)/2 * (-log(1 - v2R2)))
            end
            integral += (1 - exp(-λ * t)) * fv * dφ
        end
    end
    # Barrierless
    if φ_zb < 0.9999
        dφ2 = (0.9999 - φ_zb) / 50
        for i in 0:50
            φ = φ_zb + i * dφ2; s = 1-φ^2; s ≤ 0 && continue
            fv = exp(logC + (N-3)/2*log(s))
            integral += (1 - exp(-rate_pre * t)) * fv * dφ2
        end
    end
    return -(M-1) * integral
end

# τ from compound Poisson: find t where S(t) = 1/e
function τ_predict_cp(α, T; M=M_PAT)
    N = max(round(Int, log(M)/α), 3)
    # Binary search for t where ln S(t) = -1
    t_lo = 1.0; t_hi = 1e8
    lnS_lo = compound_poisson_lnS_single(t_lo, α, T, N, BEST_A, BEST_C; M=M, Dv_func=INTERP_DV)
    lnS_hi = compound_poisson_lnS_single(t_hi, α, T, N, BEST_A, BEST_C; M=M, Dv_func=INTERP_DV)
    lnS_lo < -1 && return t_lo  # already past
    lnS_hi > -1 && return Inf   # never reaches
    for _ in 1:60
        t_mid = sqrt(t_lo * t_hi)  # geometric bisection
        lnS_mid = compound_poisson_lnS_single(t_mid, α, T, N, BEST_A, BEST_C; M=M, Dv_func=INTERP_DV)
        if lnS_mid > -1
            t_lo = t_mid
        else
            t_hi = t_mid
        end
    end
    return sqrt(t_lo * t_hi)
end

# ══════════════════════════════════════════════════════════════════════
#                     CALIBRATION FROM v14
# ══════════════════════════════════════════════════════════════════════

# v14 τ from Poisson model fits to S(t) = exp(-K(1-exp(-t/τ_ch)))
# τ = τ_ch/K = 1/⟨λ⟩ (initial slope of disorder-averaged survival)
const v14_data = [
    (0.20, 0.25, 173281.3),
    (0.20, 0.30,  69399.2),
    (0.20, 0.40,   3345.7),
    (0.20, 0.50,    748.7),
    (0.20, 0.80,    293.6),
    (0.22, 0.20, 163040.8),
    (0.22, 0.25,  35804.7),
    (0.22, 0.30,  17968.7),
    (0.22, 0.40,   1261.6),
    (0.22, 0.50,    419.8),
    (0.22, 0.80,    231.7),
    (0.24, 0.15, 134866.0),
    (0.24, 0.20,  36113.5),
    (0.24, 0.25,  15808.7),
    (0.24, 0.30,   4486.6),
    (0.24, 0.40,    534.4),
    (0.24, 0.50,    288.3),
    (0.24, 0.80,    192.8),
    (0.26, 0.20,  14741.3),
    (0.26, 0.25,   4667.2),
    (0.26, 0.30,   1580.2),
    (0.26, 0.40,    418.9),
    (0.26, 0.50,    208.6),
    (0.28, 0.15,   9579.9),
    (0.28, 0.20,   2369.1),
    (0.28, 0.25,    935.7),
    (0.28, 0.30,    448.7),
    (0.28, 0.40,    209.6),
    (0.28, 0.50,    158.5),
]

# v8m contour points: φ=0.5 crossing → τ ≈ T_MC / ln(2 φ_eq(T))
const v8m_contour = [
    # α_cross  T
    (0.5194, 0.075),
    (0.4441, 0.125),
    (0.3889, 0.175),
    (0.3493, 0.225),
    (0.3360, 0.275),
    (0.3170, 0.325),
    (0.2996, 0.375),
    (0.2762, 0.425),
    (0.2671, 0.475),
    (0.2595, 0.525),
    (0.2561, 0.575),
    (0.2548, 0.625),
    (0.2538, 0.675),
    (0.2539, 0.725),
    (0.2549, 0.775),
    (0.2530, 0.825),
    (0.2552, 0.875),
    (0.2535, 0.925),
    (0.2541, 0.975),
    (0.2546, 1.025),
    (0.2555, 1.125),
    (0.2562, 1.225),
    (0.2558, 1.325),
    (0.2567, 1.475),
    (0.2552, 1.575),
    (0.2537, 1.675),
    (0.2520, 1.775),
    (0.2504, 1.875),
    (0.2484, 1.975),
]

println("="^60)
println("Calibrating escape formula from v14 data...")
println("  α_th (exact) = $(round(α_th, digits=4))")
println("  φ_c = $(round(φ_c, digits=4))")
println("  $(length(v14_data)) calibration points")
println("="^60)

function setup_Dv()
    # Read v13 D_v data
    v13_lines = readlines(joinpath(@__DIR__, "v13_diffusion.csv"))
    v13_α = Float64[]; v13_T = Float64[]; v13_Dv = Float64[]
    for line in v13_lines[2:end]
        f = split(line, ",")
        length(f) < 6 && continue
        push!(v13_α, parse(Float64, f[1]))
        push!(v13_T, parse(Float64, f[2]))
        push!(v13_Dv, parse(Float64, f[6]))
    end
    @printf("  Read %d v13 D_v measurements\n", length(v13_Dv))
    v13_alphas = sort(unique(v13_α))
    function interp_Dv(α_q, T_q)
        _, ia = findmin(abs.(v13_alphas .- α_q))
        α_near = v13_alphas[ia]
        mask = v13_α .== α_near
        Ts_a = v13_T[mask]; Ds_a = v13_Dv[mask]
        perm = sortperm(Ts_a); Ts_a = Ts_a[perm]; Ds_a = Ds_a[perm]
        T_q = clamp(T_q, Ts_a[1], Ts_a[end])
        for i in 1:length(Ts_a)-1
            if Ts_a[i] ≤ T_q ≤ Ts_a[i+1]
                frac = (T_q - Ts_a[i]) / (Ts_a[i+1] - Ts_a[i])
                return Ds_a[i] + frac * (Ds_a[i+1] - Ds_a[i])
            end
        end
        return Ds_a[end]
    end
    return interp_Dv
end

const INTERP_DV = setup_Dv()

# Global (A, c) from compound Poisson fit to v14 survival curves
const BEST_A = 4.5146
const BEST_C = 0.5051
@printf("  Global compound Poisson: A = %.4f, c = %.4f\n", BEST_A, BEST_C)

# ══════════════════════════════════════════════════════════════════════
#                     READ v8m HEATMAP DATA
# ══════════════════════════════════════════════════════════════════════

println("\nReading v8m data...")
v8m_path = expanduser("~/Downloads/basin_stab_LSR_v8m.csv")
lines = readlines(v8m_path)
n = length(lines) - 1
alpha_v8 = zeros(n); T_v8 = zeros(n)
phi_a = zeros(n); phi_b = zeros(n)

for i in 1:n
    f = split(lines[i+1], ",")
    alpha_v8[i] = parse(Float64, f[1])
    T_v8[i] = parse(Float64, f[2])
    phi_a[i] = parse(Float64, f[4])
    phi_b[i] = parse(Float64, f[5])
end

alphas = sort(unique(round.(alpha_v8, digits=4)))
Ts     = sort(unique(round.(T_v8, digits=4)))
na = length(alphas); nT = length(Ts)

phi_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha_v8 .- alphas[ia]) .< 0.001) .& (abs.(T_v8 .- Ts[iT]) .< 0.001)
    vals = vcat(phi_a[mask], phi_b[mask])
    !isempty(vals) && (phi_grid[iT, ia] = mean(vals))
end
@printf("  Grid: %d α × %d T\n", na, nT)

# ══════════════════════════════════════════════════════════════════════
#                     COMPUTE KRAMERS CONTOUR
# ══════════════════════════════════════════════════════════════════════

function compute_cp_contour(A_val, c_val; α_lo=0.08, α_hi=0.52, n_α=200, T_max=2.0)
    α_range_kr = range(α_lo, α_hi, length=n_α)
    α_out = Float64[]; T_out = Float64[]
    T_scan = range(0.02, T_max, length=300)
    for α_val in α_range_kr
        for i in 1:(length(T_scan)-1)
            N = max(round(Int, log(M_PAT)/α_val), 3)
            lnS1 = compound_poisson_lnS_single(Float64(T_MC_v8m), α_val, T_scan[i], N, A_val, c_val; Dv_func=INTERP_DV)
            lnS2 = compound_poisson_lnS_single(Float64(T_MC_v8m), α_val, T_scan[i+1], N, A_val, c_val; Dv_func=INTERP_DV)
            # Contour where S(T_MC) = 1/e → ln S = -1
            if lnS1 > -1 && lnS2 ≤ -1
                T_lo, T_hi = T_scan[i], T_scan[i+1]
                for _ in 1:40
                    T_mid = (T_lo + T_hi) / 2
                    lnS_mid = compound_poisson_lnS_single(Float64(T_MC_v8m), α_val, T_mid, N, A_val, c_val; Dv_func=INTERP_DV)
                    if lnS_mid > -1
                        T_lo = T_mid
                    else
                        T_hi = T_mid
                    end
                end
                push!(α_out, α_val); push!(T_out, (T_lo+T_hi)/2)
                break
            end
        end
    end
    return α_out, T_out
end

println("\nComputing contours...")
α_c0, T_c0 = compute_cp_contour(BEST_A, BEST_C)
@printf("  A=%.1f c=%.2f: %d points\n", BEST_A, BEST_C, length(α_c0))

α_c1, T_c1 = compute_cp_contour(1.0, 1.0)
@printf("  A=1.0  c=1.00: %d points\n", length(α_c1))

α_c2, T_c2 = compute_cp_contour(0.01, 0.4)
@printf("  A=0.01 c=0.40: %d points\n", length(α_c2))

# Keep original as α_contour, T_contour for the main plot
α_contour = α_c0; T_contour = T_c0

# ── P_esc = 0.63 contour from v8m (green dots for comparison) ──
# P_esc ≈ 1 - ⟨φ⟩/φ_eq(T).  At P_esc = 1-e⁻¹ ≈ 0.632: ⟨φ⟩ = e⁻¹ φ_eq
println("Computing P_esc=0.63 contour from v8m...")
α_pesc63 = Float64[]; T_pesc63 = Float64[]
for iT in 1:nT
    T_val = Ts[iT]
    φeq = φ_eq_LSR(T_val)
    target_phi = exp(-1) * φeq  # ⟨φ⟩ at P_esc = 0.63
    row = [(alphas[ia], phi_grid[iT, ia]) for ia in 1:na if !isnan(phi_grid[iT, ia])]
    for i in 1:length(row)-1
        a1, p1 = row[i]; a2, p2 = row[i+1]
        if p1 > target_phi && p2 ≤ target_phi
            frac = (target_phi - p2) / (p1 - p2)
            push!(α_pesc63, a2 + frac * (a1 - a2))
            push!(T_pesc63, T_val)
            break
        end
    end
end
@printf("  P_esc=0.63 contour: %d points\n", length(α_pesc63))

# ──────────────── K = 1 contour (exact density) ────────────────
println("Computing K=1 boundary...")
α_K1 = Float64[]
T_K1 = Float64[]
for α_val in range(α_th - 0.05, α_th + 0.10, length=50)
    K = K_exact(α_val)
    if K > 0.5  # close enough to K=1
        push!(α_K1, α_val)
    end
end
# K=1 is a vertical line at α_th (T-independent)
@printf("  α_th = %.4f\n", α_th)

# ──────────────── Exact density boundary (from ICLR paper) ────────────────
# α_c(T) = -½ ln(1 - (1 - f_ret(T))²)   where f_ret = u(φ_eq) - T s(φ_eq)
u_LSR(φ) = -log(1 - b_lsr*(1-φ)) / b_lsr
s_func(φ) = 0.5 * log(1 - φ^2)
function f_ret_LSR(T_val)
    φ = φ_eq_LSR(T_val)
    return u_LSR(φ) - T_val * s_func(φ)
end

function α_c_exact(T_val)
    fr = f_ret_LSR(T_val)
    arg = fr * (2 - fr)
    arg ≤ 0 && return Inf
    return -0.5 * log(arg)
end

# ══════════════════════════════════════════════════════════════════════
#                          PLOT
# ══════════════════════════════════════════════════════════════════════

println("\nPlotting 3-phase heatmap...")

p1 = heatmap(alphas, Ts, phi_grid,
    color=cgrad(:RdYlBu, rev=false), clims=(0, 1),
    xlabel=L"\alpha = \ln M / N", ylabel=L"T",
    xlims=(0, 0.55), ylims=(0, maximum(Ts)),
    colorbar_title=L"\langle\varphi\rangle",
    size=(FIG_W + 40, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=1Plots.mm)

# ── Phase boundaries ──

# 1. Absolute retrieval vertical line (small α)
const α_abs = 0.08
plot!(p1, [α_abs, α_abs], [0, maximum(Ts)],
      color=:blue, lw=1.5, ls=:dash, label=false)

# 2. Theoretical TD boundary (dotted): α_th vertical + α_c(T) curve
#    This is the N→∞ thermodynamic limit, NOT the main result
T_range_e = range(0.005, maximum(Ts), length=500)
T_phys_e = [t for t in T_range_e if f_ret_LSR(t) ≤ 1.0]
T_solid_e = [t for t in T_phys_e if α_c_exact(t) ≥ α_th && α_c_exact(t) ≤ 0.55]
α_solid_e = [α_c_exact(t) for t in T_solid_e]

# Find T where α_c(T) meets α_th (top of curved part)
T_join = isempty(T_solid_e) ? maximum(Ts) : maximum(T_solid_e)

# Vertical part: from T_join up to top of plot
plot!(p1, [α_th, α_th], [T_join, maximum(Ts)],
      color=:black, lw=1.5, ls=:dot,
      label=L"N \to \infty\;\mathrm{(TD)}")

# Curved part: α_c(T)
if !isempty(T_solid_e)
    plot!(p1, α_solid_e, T_solid_e,
          color=:black, lw=1.5, ls=:dot, label=false)
end

# 3. Compound Poisson contours
# Original (A=4.5, c=0.5)
if !isempty(α_c0)
    plot!(p1, α_c0, T_c0, color=:red, lw=2.0, ls=:solid,
          label=@sprintf("A=%.1f c=%.2f", BEST_A, BEST_C))
end
# c=1
if !isempty(α_c1)
    plot!(p1, α_c1, T_c1, color=:orange, lw=2.0, ls=:dash,
          label="A=1.0 c=1.0")
end
# A=500
if !isempty(α_c2)
    plot!(p1, α_c2, T_c2, color=:yellow, lw=2.0, ls=:dashdot,
          label="A=0.01 c=0.40")
end

# 4. P_esc = 0.63 from v8m (green dotted line, for comparison)
if !isempty(α_pesc63)
    plot!(p1, α_pesc63, T_pesc63,
          color=:green, lw=1.5, ls=:dot,
          label=L"P_\mathrm{esc}=0.63")
end

# Phase labels — metastable is to the LEFT of the red curve
annotate!(p1, 0.04, 1.0, text("Absolute\nretrieval", :black, 5, :center))
annotate!(p1, 0.18, 0.25, text("Metastable", :white, 6, :center))
annotate!(p1, 0.47, 0.8, text("Non-\nretrieval", :white, 6, :center))

# v14 calibration points
for (α, T, τm) in v14_data
    scatter!(p1, [α], [T], markersize=2, color=:white,
             markerstrokecolor=:black, markerstrokewidth=0.5, label=false)
end

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "heatmap_3phase.$ext"))
end
println("Saved: panels_paper/heatmap_3phase.{png,pdf}")

println("\n" * "="^60)
@printf("Summary:\n")
@printf("  α_th (exact) = %.4f = ln(2)/2\n", α_th)
@printf("  A = %.4f, c = %.4f (compound Poisson)\n", BEST_A, BEST_C)
println("="^60)

#=
Phase diagram heatmaps from v10 MC data with theory overlays
────────────────────────────────────────────────────────────────────────
Output: panels_v10/phase_LSE_v10.{png,pdf}
        panels_v10/phase_LSR_v10.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using Statistics

# ──────────────── Figure settings (86mm) ────────────────
const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = FIG_W
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_ANN   = 7

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = "panels_v10"
mkpath(out_dir)

# ──────────────── LSR theory ────────────────
const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr

# LSR retrieval overlap (ICML Eq. 37)
function φ_LSR(T)
    a = b_lsr*T + 1; bc = -(2 + T + T*b_lsr); c = T
    disc = bc^2 - 4*a*c; disc < 0 && return NaN
    return 1 - (-bc - sqrt(disc)) / (2*a)
end

u_LSR(φ) = -log(1 - b_lsr*(1-φ)) / b_lsr
s(φ) = 0.5 * log(1 - φ^2)
f_ret_LSR(T) = let φ = φ_LSR(T); u_LSR(φ) - T*s(φ); end

α_th_gauss = φ_c^2 / 2
α_c_gauss_LSR(T) = 0.5 * (1 - f_ret_LSR(T))^2
α_c_exact_LSR(T) = let fr = f_ret_LSR(T); arg = fr*(2-fr); arg <= 0 ? Inf : -0.5*log(arg); end

# ──────────────── LSE theory ────────────────
φ_LSE(T) = 0.5 * (-T + sqrt(T^2 + 4))
f_ret_LSE(T) = let φ = φ_LSE(T); 1 - φ - (T/2)*log(1 - φ^2); end
α_c_gauss_LSE(T) = 0.5 * (1 - f_ret_LSE(T))^2
α_c_exact_LSE(T) = let fr = f_ret_LSE(T); arg = fr*(2-fr); arg <= 0 ? Inf : -0.5*log(arg); end

# ──────────────── Read CSV ────────────────
function read_v10(filename)
    lines = readlines(filename)
    n = length(lines) - 1
    alpha = zeros(n); T = zeros(n); phi_a = zeros(n); phi_b = zeros(n)
    for i in 1:n
        f = split(lines[i+1], ",")
        alpha[i] = parse(Float64, f[1])
        T[i] = parse(Float64, f[2])
        phi_a[i] = parse(Float64, f[4])
        phi_b[i] = parse(Float64, f[5])
    end
    return alpha, T, phi_a, phi_b
end

function make_grid(alpha, T, phi_a, phi_b)
    alphas = sort(unique(round.(alpha, digits=4)))
    Ts = sort(unique(round.(T, digits=5)))
    na = length(alphas); nT = length(Ts)
    grid = zeros(nT, na)
    for ia in 1:na, iT in 1:nT
        mask = (abs.(alpha .- alphas[ia]) .< 0.001) .& (abs.(T .- Ts[iT]) .< 0.0001)
        vals = vcat(phi_a[mask], phi_b[mask])
        grid[iT, ia] = isempty(vals) ? NaN : mean(vals)
    end
    return alphas, Ts, grid
end

# ──────────────── Plot function ────────────────
function plot_phase(alphas, Ts, grid, theory_gauss, theory_exact, α_th_g, α_th_e, title_str, filename)
    p = heatmap(alphas, Ts, grid,
        color=:RdBu, clims=(0, 1),
        xlabel="α", ylabel="T",
        xlims=(0, 1.0), ylims=(0, 1.0),
        colorbar_title="⟨φ⟩",
        size=(FIG_W + 40, FIG_H), dpi=FIG_DPI,
        left_margin=0Plots.mm, bottom_margin=0Plots.mm,
        )

    # Gaussian theory curve
    T_th = range(0.005, 1.0, length=300)
    α_g = [theory_gauss(T) for T in T_th]
    α_e = [min(theory_exact(T), 1.05) for T in T_th]

    # For LSR: only physical branch (f_ret ≤ 1)
    if α_th_g !== nothing
        # Gaussian: vertical + curve (only where α > α_th and physical)
        T_solid = [T for T in T_th if theory_gauss(T) >= α_th_g && theory_gauss(T) <= 1.0 &&
                   (title_str[1:3] == "LSE" || f_ret_LSR(T) <= 1.0)]
        α_solid = [theory_gauss(T) for T in T_solid]
        T_dot = [T for T in T_th if theory_gauss(T) < α_th_g &&
                 (title_str[1:3] == "LSE" || f_ret_LSR(T) <= 1.0)]
        α_dot = [theory_gauss(T) for T in T_dot]

        T_join = isempty(T_solid) ? 0.0 : maximum(T_solid)
        plot!(p, [α_th_g, α_th_g], [T_join, 1.0], color=:blue, lw=2, label=false)
        !isempty(T_solid) && plot!(p, α_solid, T_solid, color=:blue, lw=2, label="Gaussian")
        !isempty(T_dot) && plot!(p, α_dot, T_dot, color=:blue, lw=1.5, ls=:dot, label=false)

        # Exact: vertical + curve
        α_th_exact = α_th_e
        T_solid_e = [T for T in T_th if theory_exact(T) >= α_th_exact && theory_exact(T) <= 1.0 &&
                     (title_str[1:3] == "LSE" || f_ret_LSR(T) <= 1.0)]
        α_solid_e = [theory_exact(T) for T in T_solid_e]
        T_dot_e = [T for T in T_th if theory_exact(T) < α_th_exact &&
                   (title_str[1:3] == "LSE" || f_ret_LSR(T) <= 1.0)]
        α_dot_e = [min(theory_exact(T), 1.0) for T in T_dot_e]

        T_join_e = isempty(T_solid_e) ? 0.0 : maximum(T_solid_e)
        plot!(p, [α_th_exact, α_th_exact], [T_join_e, 1.0], color=:red, lw=2, ls=:dash, label=false)
        !isempty(T_solid_e) && plot!(p, α_solid_e, T_solid_e, color=:red, lw=2, ls=:dash, label="Exact")
        !isempty(T_dot_e) && plot!(p, α_dot_e, T_dot_e, color=:red, lw=1.5, ls=:dot, label=false)
    else
        # LSE: no vertical threshold, just curves
        T_g = [T for T in T_th if theory_gauss(T) <= 1.0]
        plot!(p, [theory_gauss(T) for T in T_g], T_g, color=:blue, lw=2, label="Gaussian")
        T_e = [T for T in T_th if theory_exact(T) <= 1.0]
        plot!(p, [min(theory_exact(T), 1.0) for T in T_e], T_e, color=:red, lw=2, ls=:dash, label="Exact")
    end

    for ext in ("png", "pdf")
        savefig(p, joinpath(out_dir, "$filename.$ext"))
    end
    println("Saved: $out_dir/$filename.{png,pdf}")
end

# ──────────────── LSE ────────────────
println("Reading LSE v10...")
α_lse, T_lse, pa_lse, pb_lse = read_v10("NIPS_Resilience/code/basin_stab_LSE_v10.csv")
alphas_lse, Ts_lse, grid_lse = make_grid(α_lse, T_lse, pa_lse, pb_lse)
plot_phase(alphas_lse, Ts_lse, grid_lse,
    α_c_gauss_LSE, α_c_exact_LSE, nothing, nothing,
    "LSE (v10)", "phase_LSE_v10")

# ──────────────── LSR ────────────────
println("Reading LSR v10...")
α_lsr, T_lsr, pa_lsr, pb_lsr = read_v10("NIPS_Resilience/code/basin_stab_LSR_v10.csv")
alphas_lsr, Ts_lsr, grid_lsr = make_grid(α_lsr, T_lsr, pa_lsr, pb_lsr)

α_th_exact_lsr = -0.5 * log(1 - φ_c^2)
plot_phase(alphas_lsr, Ts_lsr, grid_lsr,
    α_c_gauss_LSR, α_c_exact_LSR, α_th_gauss, α_th_exact_lsr,
    "LSR (v10, b=3.41)", "phase_LSR_v10")

println("\nDone.")

#=
LSE per-trial φ CDFs from v8m_a1 (AAAI paper)
────────────────────────────────────────────────────────────────────────
Mirrors plot_cdf_v8m.jl structure, but for the LSE energy with
α ∈ [0.01, 1.00].  Two panels:
  Panel A — fixed T = 0.025 (lowest), curves at varying α
            (largest Gaussian-vs-exact split is at T → 0)
  Panel B — fixed α = 0.50 (Gaussian boundary at T=0), varying T

Input : basin_stab_LSE_v8m_a1.csv
Output: ../panels_paper/cdf_LSE_panel_A.{png,pdf}
        ../panels_paper/cdf_LSE_panel_B.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# ──────────────── Configuration ────────────────
csv_file = joinpath(@__DIR__, "basin_stab_LSE_v8m_a1.csv")
out_dir  = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# Panel A: fixed T_A varying α (T_A = lowest grid value)
const T_A = 0.025
const PROBE_A = [
    (0.10, T_A),
    (0.30, T_A),
    (0.50, T_A),   # Gaussian boundary at T=0
    (0.70, T_A),
    (0.90, T_A),
]

# Panel B: fixed α_B varying T (α_B sits on Gaussian boundary at T=0)
const α_B = 0.50
const PROBE_B = [
    (α_B, 0.025),
    (α_B, 0.125),
    (α_B, 0.325),
    (α_B, 0.525),
    (α_B, 0.825),
    (α_B, 1.225),
]

# ──────────────── Figure settings (match plot_cdf_v8m.jl) ────────────────
const FIG_DPI    = 300
const FIG_W_STD  = round(Int, 86 / 25.4 * 100)   # 339 (standard 86mm)
const FIG_W      = 2 * FIG_W_STD                  # 678
const FIG_H      = round(Int, FIG_W_STD * 0.85)   # 288

const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 6

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND)

# ──────────────── LSE equilibrium overlap ────────────────
φ_eq_LSE(T) = 0.5 * (-T + sqrt(T^2 + 4))

# ──────────────── Read data ────────────────
println("Reading $csv_file ...")
df = CSV.read(csv_file, DataFrame)
alpha_all = sort(unique(df.alpha))
T_all     = sort(unique(df.T))
@printf("Loaded: %d rows, %d α × %d T\n", nrow(df), length(alpha_all), length(T_all))

# ──────────────── Snap to closest grid point ────────────────
function closest(vec, val)
    _, idx = findmin(abs.(vec .- val))
    return vec[idx]
end

# ──────────────── Panel plotter ────────────────
function plot_cdf_panel(probes, title_str, fname; legend_loc=:outertopright)
    p = plot(xlabel="φ", ylabel="CDF",
        title=title_str,
        legend=legend_loc, legendfontsize=FONT_LEGEND,
        size=(FIG_W, FIG_H), dpi=FIG_DPI,
        xlims=(-0.05, 1.05), ylims=(0, 1),
        right_margin=3Plots.mm, left_margin=3Plots.mm,
        top_margin=3Plots.mm, bottom_margin=3Plots.mm)

    # Colors interpolate from blue (low α/T) to red (high α/T)
    nprobe = length(probes)
    colors = cgrad(:RdYlBu, nprobe, rev=true)

    for (i, (α_target, T_target)) in enumerate(probes)
        α = closest(alpha_all, α_target)
        T = closest(T_all, T_target)
        sub = df[(df.alpha .== α) .& (df.T .== T), :]
        phis = sort(vcat(sub.phi_a, sub.phi_b))
        n = length(phis)
        n == 0 && (println("  no data at α=$α T=$T"); continue)
        cdf_y = (1:n) ./ n

        φeq = φ_eq_LSE(T)
        φ_mean = mean(phis)
        lab = @sprintf("α=%.2f T=%.3f  ⟨φ⟩=%.2f φ_eq=%.2f n=%d",
                       α, T, φ_mean, φeq, n)
        plot!(p, phis, cdf_y, lw=1.5, color=colors[i], label=lab)
    end

    # Reference: φ_eq at the (fixed) condition shared by the panel
    Tref_set = unique([T for (_, T) in probes])
    αref_set = unique([a for (a, _) in probes])
    if length(Tref_set) == 1
        vline!(p, [φ_eq_LSE(Tref_set[1])],
               color=:gray, ls=:dash, lw=0.8,
               label=@sprintf("φ_eq(T=%.3f)=%.3f",
                              Tref_set[1], φ_eq_LSE(Tref_set[1])))
    end

    for ext in ("png", "pdf")
        savefig(p, joinpath(out_dir, "$fname.$ext"))
    end
    println("Saved: panels_paper/$fname.{png,pdf}")
end

# ──────────────── Panel A: fixed T, varying α ────────────────
plot_cdf_panel(PROBE_A,
               @sprintf("LSE  v8m_a1   fixed T = %.3f", T_A),
               "cdf_LSE_panel_A")

# ──────────────── Panel B: fixed α, varying T ────────────────
plot_cdf_panel(PROBE_B,
               @sprintf("LSE  v8m_a1   fixed α = %.2f", α_B),
               "cdf_LSE_panel_B")

# ──────────────── Summary stats ────────────────
println("\nSummary at T = $T_A (lowest grid T, where Gaussian/exact split is largest):")
@printf("  %-5s  %5s  %6s  %6s  %6s\n", "α", "n_dis", "⟨φ⟩", "σ(φ)", "frac<0.5")
println("  " * "─"^45)
φ_eq_A = φ_eq_LSE(T_A)
for α_target in [0.10, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    α = closest(alpha_all, α_target)
    T = closest(T_all, T_A)
    sub = df[(df.alpha .== α) .& (df.T .== T), :]
    phis = vcat(sub.phi_a, sub.phi_b)
    isempty(phis) && continue
    @printf("  %.2f   %4d  %6.3f  %5.3f    %5.3f\n",
            α, length(sub.phi_a), mean(phis), std(phis), mean(phis .< 0.5))
end

@printf("\nReference: α_c^G(0) = 0.500   α_c^E(0) = ∞   φ_eq(%.3f) = %.4f\n",
        T_A, φ_eq_A)
println("\nDone.")

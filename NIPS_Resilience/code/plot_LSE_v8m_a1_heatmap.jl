#=
LSE phase diagram heatmap from v8m_a1 bulk survey (AAAI paper)
────────────────────────────────────────────────────────────────────────
Mirrors plot_section3_heatmap.jl structure, but for the LSE energy
with α ∈ [0.01, 1.00].  Shows ⟨φ⟩(α,T) with Gaussian and exact
theoretical boundaries overlaid.

LSE theory (cf. latex/LSE_gaussian.tex):
  φ_eq(T)   = (1/2)(-T + √(T²+4))
  f_ret(T)  = 1 - φ_eq - (T/2) ln(1 - φ_eq²)
  α_c^G(T)  = (1/2)(1 - f_ret)²                  Gaussian   → 0.5 at T=0
  α_c^E(T)  = -(1/2) ln(1 - (1-f_ret)²)          Exact      → ∞   at T=0

Input : basin_stab_LSE_v8m_a1.csv
Output: ../panels_paper/heatmap_LSE.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

ENV["GKSwstype"] = "100"   # file-only GR backend (headless: no display)

using Plots
using Printf
using Statistics

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

# ──────────────── Read data ────────────────
csv_in = joinpath(@__DIR__, "basin_stab_LSE_v8m_a1.csv")
println("Reading $csv_in ...")
lines = readlines(csv_in)
n = length(lines) - 1
alpha = zeros(n); T = zeros(n); phi_a = zeros(n); phi_b = zeros(n)
for i in 1:n
    f = split(lines[i+1], ",")
    alpha[i] = parse(Float64, f[1]); T[i] = parse(Float64, f[2])
    phi_a[i] = parse(Float64, f[4]); phi_b[i] = parse(Float64, f[5])
end

alphas = sort(unique(round.(alpha, digits=4)))
Ts     = sort(unique(round.(T,     digits=4)))
na = length(alphas); nT = length(Ts)
@printf("Loaded: %d rows, %d α × %d T\n", n, na, nT)

# ──────────────── ⟨φ⟩ grid ────────────────
phi_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 0.001) .& (abs.(T .- Ts[iT]) .< 0.001)
    vals = vcat(phi_a[mask], phi_b[mask])
    !isempty(vals) && (phi_grid[iT, ia] = mean(vals))
end

# ──────────────── LSE theory ────────────────
φ_eq_LSE(T)  = 0.5 * (-T + sqrt(T^2 + 4))
s_func(φ)    = 0.5 * log(1 - φ^2)
f_ret_LSE(T) = let φ = φ_eq_LSE(T); 1 - φ - T*s_func(φ); end

α_c_gauss(T) = let fr = f_ret_LSE(T); fr >= 1 ? 0.0 : 0.5 * (1 - fr)^2; end
function α_c_exact(T)
    fr = f_ret_LSE(T)
    fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5 * log(arg)
end

# ──────────────── PLOT HEATMAP ────────────────
println("Plotting heatmap...")

p1 = heatmap(alphas, Ts, phi_grid,
    color=cgrad(:RdYlBu, rev=false), clims=(0, 1),
    xlabel="α = ln M / N", ylabel="T",
    xlims=(0, 1.0), ylims=(0, maximum(Ts)),
    colorbar_title="⟨φ⟩",
    size=(FIG_W + 40, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=1Plots.mm)

# Theory overlays
T_range = range(0.005, maximum(Ts), length=500)

# Gaussian: defined for all T > 0; → 0.5 as T → 0
α_g = [α_c_gauss(t) for t in T_range]
mask_g = (α_g .>= 0) .& (α_g .<= 1.0)
plot!(p1, α_g[mask_g], T_range[mask_g],
      color=:blue, lw=2.0, ls=:solid,
      label="α_c^G(T) Gaussian")

# Exact: diverges as T → 0; clip at the right edge of the plot
α_e = [α_c_exact(t) for t in T_range]
mask_e = isfinite.(α_e) .& (α_e .<= 1.0)
plot!(p1, α_e[mask_e], T_range[mask_e],
      color=:red, lw=2.0, ls=:dash,
      label="α_c^E(T) Exact")

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "heatmap_LSE.$ext"))
end
println("Saved: panels_paper/heatmap_LSE.{png,pdf}")

println("\nDone.")

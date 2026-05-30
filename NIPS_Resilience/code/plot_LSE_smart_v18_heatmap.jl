#=
Smart-MC LSE heatmap (v18) — AAAI paper
────────────────────────────────────────────────────────────────────────
Reads basin_stab_LSE_smart_v18_N<N>_K<K>.csv and produces a (α, T)
heatmap of ⟨φ⟩, with overlays of:

  • α_c^E(T)  exact LSE retrieval boundary  (red, dashed)
  • α_c^G(T)  Gaussian (Ramsauer+)          (blue, solid)
  • T_max(α)  smart-MC validity wall        (grey, dotted)

Usage:
  julia plot_LSE_smart_v18_heatmap.jl N K
  (defaults: N=100, K=10000)

Output: ../panels_paper/heatmap_LSE_smart_v18_N<N>_K<K>.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

ENV["GKSwstype"] = "100"

using Plots
using Printf
using Statistics
using SpecialFunctions
using QuadGK

const N_ARG = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
const K_ARG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000

const FIG_DPI    = 300
const FIG_W      = round(Int, 86 / 25.4 * 100) + 60
const FIG_H      = round(Int, 86 / 25.4 * 100)
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_LEG   = 6
default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK, legendfontsize=FONT_LEG)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

csv_in = joinpath(@__DIR__,
    @sprintf("basin_stab_LSE_smart_v18_N%d_K%d.csv", N_ARG, K_ARG))
isfile(csv_in) || error("Missing $csv_in — run basin_stab_LSE_smart_v18.jl first")

# ──────────────── Read CSV ────────────────
println("Reading $csv_in ...")
lines = filter(l -> !startswith(l, "#") && !isempty(l), readlines(csv_in))
header = lines[1]; lines = lines[2:end]
n = length(lines)
alpha = zeros(n); T = zeros(n); phi_a = zeros(n); phi_b = zeros(n)
for i in 1:n
    f = split(lines[i], ",")
    alpha[i] = parse(Float64, f[1]); T[i] = parse(Float64, f[2])
    phi_a[i] = parse(Float64, f[4]); phi_b[i] = parse(Float64, f[5])
end
alphas = sort(unique(round.(alpha, digits=4)))
@printf("Rows: %d   α values: %d   α∈[%.3f, %.3f]\n",
        n, length(alphas), first(alphas), last(alphas))

# Wedge grid: T values differ per α.  Build a regular T-axis for the heatmap
# by union-of-grids + nearest-snap; cells where the sim never visited (above
# T_max(α)) will be NaN and the heatmap shows them as background.
T_all_set = Set{Float64}()
for ti in T; push!(T_all_set, round(ti, digits=5)); end
Ts = sort(collect(T_all_set))
na = length(alphas); nT = length(Ts)
phi_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 1e-4) .& (abs.(T .- Ts[iT]) .< 1e-4)
    vals = vcat(phi_a[mask], phi_b[mask])
    !isempty(vals) && (phi_grid[iT, ia] = mean(vals))
end

# ──────────────── Theory ────────────────
φ_eq(T)       = 0.5*(-T + sqrt(T^2 + 4))
f_ret(T)      = let φ = φ_eq(T); 1 - φ - (T/2)*log(1 - φ^2); end
α_c_gauss(T)  = let fr = f_ret(T); fr >= 1 ? 0.0 : 0.5*(1 - fr)^2; end
α_c_exact(T)  = let fr = f_ret(T)
    fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5*log(arg)
end

# Smart-MC validity wall: T_max(α, N, K) = (1-φ_cut²)/φ_cut
const beta_a = (N_ARG - 1)/2

function log_beta_sf(x, a)
    x <= 0 && return 0.0; x >= 1 && return -Inf
    if x < 0.95
        _, q = beta_inc(a, a, x); q > 0 && return log(q)
    end
    u = 1 - x
    integrand(t) = t > 0 ? exp((a-1)*(log(t) + log1p(-u*t))) : 0.0
    val, _ = quadgk(integrand, 0.0, 1.0; rtol=1e-10)
    return a*log(u) + log(val) - logbeta(a, a)
end
function beta_isf_log(log_p, a)
    log_p >= 0 && return 0.0
    log_u = (log_p + log(a) + logbeta(a, a)) / a
    log_u = min(log_u, log(0.5))
    for _ in 1:60
        u = clamp(exp(log_u), 1e-300, 0.5); x = 1 - u
        lp = log_beta_sf(x, a); err = lp - log_p
        abs(err) < 1e-10 && break
        log_pdf = (a-1)*(log(x) + log(u)) - logbeta(a, a)
        dlog = exp(log_pdf + log(u) - lp); dlog < 1e-12 && (dlog = a)
        log_u -= err / dlog
    end
    return 1 - exp(log_u)
end
function phi_cut(α)
    log_p = log(K_ARG) - α*N_ARG
    log_p >= 0 && return -1.0
    return 2*beta_isf_log(log_p, Float64(beta_a)) - 1
end
T_max(α) = let φc = phi_cut(α); φc <= 0 ? Inf : (1 - φc^2)/φc; end

# ──────────────── Plot ────────────────
println("Plotting heatmap...")

xmin, xmax = first(alphas), last(alphas)
ymin, ymax = minimum(Ts), maximum(Ts) * 1.05

p1 = heatmap(alphas, Ts, phi_grid,
    color=cgrad(:RdYlBu, rev=false), clims=(0, 1),
    xlabel="α = ln M / N", ylabel="T",
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    colorbar_title="⟨φ⟩",
    title=@sprintf("Smart-MC LSE   N=%d  K=%d", N_ARG, K_ARG),
    titlefontsize=FONT_GUIDE,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=2Plots.mm)

# Theory overlays
T_range = range(max(ymin, 1e-4), ymax, length=600)

α_g = [α_c_gauss(t) for t in T_range]
mask_g = (α_g .>= xmin) .& (α_g .<= xmax)
plot!(p1, α_g[mask_g], T_range[mask_g],
      color=:blue, lw=2.0, ls=:solid, label="α_c^G(T)  Gaussian")

α_e = [α_c_exact(t) for t in T_range]
mask_e = isfinite.(α_e) .& (α_e .>= xmin) .& (α_e .<= xmax)
plot!(p1, α_e[mask_e], T_range[mask_e],
      color=:red, lw=2.0, ls=:dash, label="α_c^E(T)  Exact")

# Smart-MC validity wall (T_max as a function of α)
α_wall = collect(range(xmin, xmax, length=300))
T_wall = [T_max(α) for α in α_wall]
mask_w = isfinite.(T_wall) .& (T_wall .<= ymax)
plot!(p1, α_wall[mask_w], T_wall[mask_w],
      color=:black, lw=1.0, ls=:dot, label="T_max(α,N,K)  validity wall")

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir,
        @sprintf("heatmap_LSE_smart_v18_N%d_K%d.%s", N_ARG, K_ARG, ext)))
end
@printf("Saved: panels_paper/heatmap_LSE_smart_v18_N%d_K%d.{png,pdf}\n",
        N_ARG, K_ARG)
println("\nDone.")

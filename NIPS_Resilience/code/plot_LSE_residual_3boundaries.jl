#=
Honest N=25 LSE residual heatmap with three analytical α_c(T) boundaries
overlaid for direct comparison:

  (1) α_c^Gauss(T)     = ½·(1 − f_ret(T))²
        — paper Eq. 35; Gaussian-density boundary regime.
        T=0 limit: α_c = 0.5 (classical Ramsauer capacity).

  (2) α_c^bd(T)        = −½·ln[ 1 − (1 − f_ret(T))² ]
        — exact-sphere density, but applied via the boundary (φ_max) regime.
        Valid only for α < 0.241 where the integrand saddle is outside
        the available φ range. T=0 limit: α_c → ∞.

  (3) α_c^sd(T)        = 0.6226 − f_ret(T)
        — exact-sphere density, saddle-dominated regime (the integrand of
        ∫dφ·M·ρ(φ)·exp(−β_net·N·(1−φ)) is peaked at φ* = (√5−1)/2 ≈ 0.618).
        Valid for α > 0.241 (the entire relevant range here).
        T=0 limit: α_c = 1 − φ*² + ½·log(φ*) … numerically 0.6226 (golden).

φ_eq(T)   = ½(−T + √(T²+4))                              (icml Eq. 33)
f_ret(T)  = 1 − φ_eq − (T/2)·log(1 − φ_eq²)              (icml Eq. 34)

Source heatmap: basin_stab_LSE_honest_N25.csv.

Output: ../panels_paper/heatmap_LSE_honest_N25_residual_3boundaries.{png,pdf}
=#

ENV["GKSwstype"] = "100"
using Plots, Printf, Statistics

const N_SRC = 25

const FIG_DPI    = 300
const FIG_W      = round(Int, 86 / 25.4 * 100) + 60
const FIG_H      = round(Int, 86 / 25.4 * 100)
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_LEG   = 6
default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK, legendfontsize=FONT_LEG)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# ─── Load honest N=25 data ───
csv_in = joinpath(@__DIR__, @sprintf("basin_stab_LSE_honest_N%d.csv", N_SRC))
isfile(csv_in) || error("Missing $csv_in")
println("Reading $csv_in ...")
lines = filter(l -> !startswith(l, "#") && !isempty(l), readlines(csv_in))
lines = lines[2:end]
n = length(lines)
alpha = zeros(n); T = zeros(n); phi_a = zeros(n); phi_b = zeros(n)
for i in 1:n
    f = split(lines[i], ",")
    alpha[i] = parse(Float64, f[1]); T[i] = parse(Float64, f[2])
    phi_a[i] = parse(Float64, f[4]); phi_b[i] = parse(Float64, f[5])
end
alphas = sort(unique(round.(alpha, digits=4)))
Ts     = sort(unique(round.(T,     digits=5)))
na = length(alphas); nT = length(Ts)
@printf("Rows: %d   α values: %d   T values: %d\n", n, na, nT)

φ_eq(t) = 0.5 * (-t + sqrt(t^2 + 4))
phi_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 1e-4) .& (abs.(T .- Ts[iT]) .< 1e-5)
    vals = vcat(phi_a[mask], phi_b[mask])
    !isempty(vals) && (phi_grid[iT, ia] = mean(vals))
end
res_grid = similar(phi_grid)
for iT in 1:nT
    base = φ_eq(Ts[iT])
    res_grid[iT, :] .= phi_grid[iT, :] .- base
end
finite_res = filter(isfinite, vec(res_grid))
cmax = max(0.05, maximum(abs, finite_res))

# ─── Three analytical boundaries ───
f_ret(t)         = let φ = φ_eq(t); 1 - φ - (t/2)*log(1 - φ^2); end
α_c_gauss(t)     = let fr = f_ret(t); fr >= 1 ? 0.0 : 0.5*(1 - fr)^2; end
function α_c_bd(t)                                  # boundary (current "exact" in code)
    fr = f_ret(t); fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5*log(arg)
end
const φ_star = (sqrt(5.0) - 1)/2                    # golden ratio
# g_max = ½ log(1-φ*²) − (1−φ*); using 1-φ*² = φ* and 1-φ* = φ*²:
const G_MAX  = 0.5*log(φ_star) - φ_star^2          # = -0.62258...
α_c_sd(t)        = -G_MAX - f_ret(t)                # saddle-dominated correct exact

# ─── Plot ───
println("Plotting...")
xmin, xmax = first(alphas) - 0.01, last(alphas) + 0.01
ymin, ymax = 0.0, 0.5

p1 = heatmap(alphas, Ts, res_grid,
    color=cgrad(:RdBu), clims=(-cmax, cmax),
    xlabel="α  =  ln M / N", ylabel="T",
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    colorbar_title="⟨φ⟩ − φ_eq(T)",
    title=@sprintf("Honest LSE residual N=%d  +  3 analytical α_c(T)", N_SRC),
    titlefontsize=FONT_GUIDE,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=2Plots.mm)

T_range = range(max(ymin, 1e-4), ymax, length=600)

# (1) Gaussian (Ramsauer): blue solid
α_g = [α_c_gauss(t) for t in T_range]
m_g = isfinite.(α_g) .& (α_g .>= xmin) .& (α_g .<= xmax)
plot!(p1, α_g[m_g], T_range[m_g],
      color=:blue, lw=2.0, ls=:solid, label="α_c^Gauss  (Ramsauer)")

# (2) Boundary "exact" (current code's α_c^E): orange dashed
α_b = [α_c_bd(t) for t in T_range]
m_b = isfinite.(α_b) .& (α_b .>= xmin) .& (α_b .<= xmax)
plot!(p1, α_b[m_b], T_range[m_b],
      color=:darkorange, lw=2.0, ls=:dash, label="α_c^exact  boundary form")

# (3) Saddle "exact" (correct): red solid
α_s = [α_c_sd(t) for t in T_range]
m_s = isfinite.(α_s) .& (α_s .>= xmin) .& (α_s .<= xmax)
plot!(p1, α_s[m_s], T_range[m_s],
      color=:red, lw=2.0, ls=:solid, label="α_c^exact  saddle (correct)")

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir,
        @sprintf("heatmap_LSE_honest_N%d_residual_3boundaries.%s", N_SRC, ext)))
end
@printf("Saved: panels_paper/heatmap_LSE_honest_N%d_residual_3boundaries.{png,pdf}\n", N_SRC)
@printf("φ* = %.6f   g_max = %.6f   so α_c^sd(T=0) = %.4f\n",
        φ_star, G_MAX, -G_MAX)
println("Done.")

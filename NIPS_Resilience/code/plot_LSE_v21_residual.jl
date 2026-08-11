#=
Residual heatmap for v21 CPU prototype.
Δ(α, T) = ⟨φ⟩(α, T) − φ_eq(T), diverging RdBu, white at 0.

Usage: julia plot_LSE_v21_residual.jl N K
Output: ../panels_paper/heatmap_LSE_v21_cpu_N<N>_K<K>_residual.{png,pdf}
=#
ENV["GKSwstype"] = "100"
using Plots, Printf, Statistics

const N_ARG = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
const K_ARG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000

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
    @sprintf("basin_stab_LSE_v21_cpu_N%d_K%d.csv", N_ARG, K_ARG))
isfile(csv_in) || error("Missing $csv_in — run basin_stab_LSE_v21_cpu.jl first")

println("Reading $csv_in ...")
lines = filter(l -> !startswith(l, "#") && !isempty(l), readlines(csv_in))
lines = lines[2:end]
n = length(lines)
alpha = zeros(n); T = zeros(n); phi = zeros(n)
for i in 1:n
    f = split(lines[i], ",")
    alpha[i] = parse(Float64, f[1])
    T[i]     = parse(Float64, f[2])
    phi[i]   = parse(Float64, f[4])
end
alphas = sort(unique(round.(alpha, digits=4)))
Ts     = sort(unique(round.(T,     digits=5)))
na = length(alphas); nT = length(Ts)
@printf("Rows: %d   α values: %d   T values: %d\n", n, na, nT)

φ_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

phi_grid = fill(NaN, nT, na)
for ia in 1:na, iT in 1:nT
    mask = (abs.(alpha .- alphas[ia]) .< 1e-4) .& (abs.(T .- Ts[iT]) .< 1e-5)
    any(mask) && (phi_grid[iT, ia] = mean(phi[mask]))
end
res_grid = similar(phi_grid)
for iT in 1:nT
    base = φ_eq(Ts[iT])
    res_grid[iT, :] .= phi_grid[iT, :] .- base
end
finite_res = filter(isfinite, vec(res_grid))
cmax = max(0.05, maximum(abs, finite_res))
@printf("Residual range: [%.3f, %.3f]   colour limits: ±%.2f\n",
        minimum(finite_res), maximum(finite_res), cmax)

f_ret(T)     = let φ = φ_eq(T); 1 - φ - (T/2)*log(1 - φ^2); end
α_c_gauss(T) = let fr = f_ret(T); fr >= 1 ? 0.0 : 0.5*(1 - fr)^2; end
function α_c_exact(T)
    fr = f_ret(T); fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5*log(arg)
end

println("Plotting residual heatmap...")
xmin, xmax = first(alphas) - 0.01, last(alphas) + 0.01
ymin, ymax = 0.0, 0.5

p1 = heatmap(alphas, Ts, res_grid,
    color=cgrad(:RdBu), clims=(-cmax, cmax),
    xlabel="α  =  ln M / N", ylabel="T",
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    colorbar_title="⟨φ⟩ − φ_eq(T)",
    title=@sprintf("v21 sticky-cache  N=%d  K=%d", N_ARG, K_ARG),
    titlefontsize=FONT_GUIDE,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=2Plots.mm)

T_range = range(max(ymin, 1e-4), ymax, length=600)
α_g = [α_c_gauss(t) for t in T_range]
mask_g = (α_g .>= xmin) .& (α_g .<= xmax)
plot!(p1, α_g[mask_g], T_range[mask_g],
      color=:blue, lw=2.0, ls=:solid, label="α_c^G(T)  Gaussian")
α_e = [α_c_exact(t) for t in T_range]
mask_e = isfinite.(α_e) .& (α_e .>= xmin) .& (α_e .<= xmax)
plot!(p1, α_e[mask_e], T_range[mask_e],
      color=:red, lw=2.0, ls=:dash, label="α_c^E(T)  Exact")

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir,
        @sprintf("heatmap_LSE_v21_cpu_N%d_K%d_residual.%s", N_ARG, K_ARG, ext)))
end
@printf("Saved: panels_paper/heatmap_LSE_v21_cpu_N%d_K%d_residual.{png,pdf}\n",
        N_ARG, K_ARG)

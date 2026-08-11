#=
Take the honest N=25 LSE residual heatmap and overlay self-averaged V_eff
boundary curves at N=25, 100, 500, along with the analytical α_c^E and α_c^G
limits.

Self-averaged effective potential (1D in m = φ_x¹):
    V_eff(m) = -(1/βn) · log[ exp(βn·N·m) + M · μ̄(N, βn) ]
where
    μ̄(N, βn) = E_ξ[exp(βn·ξ·x)]
              = ∫_{-1}^{1} (1-u²)^{(N-3)/2}/B((N-1)/2, 1/2) · exp(βn·N·u) du
is the expected single-pattern Boltzmann factor (rotationally x-independent).

Marginal sphere measure for m: (1-m²)^{(N-3)/2}.

P(m; α, T, N) ∝ exp[-β·V_eff(m)] · (1-m²)^{(N-3)/2}
⟨φ⟩(α, T, N)  = ∫ m·P / ∫ P

Boundary α*(T, N) = α at which ⟨φ⟩ = 0.5·φ_eq(T) (bisection in α).
This is a finite-N approximation of α_c^E(T); converges to it as N → ∞.

Usage:  julia plot_LSE_residual_with_Veff_overlay.jl
Output: ../panels_paper/heatmap_LSE_honest_N25_residual_Veff_overlay.{png,pdf}
=#

ENV["GKSwstype"] = "100"

using Plots, Printf, Statistics
using QuadGK, SpecialFunctions

const N_SRC      = 25       # source heatmap N
const βn         = 1.0
const NS_OVERLAY = [25, 100, 500]
const THR_RATIO  = 0.5      # ⟨φ⟩ / φ_eq at boundary

const FIG_DPI    = 300
const FIG_W      = round(Int, 86 / 25.4 * 100) + 60
const FIG_H      = round(Int, 86 / 25.4 * 100)
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_LEG   = 6
default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK, legendfontsize=FONT_LEG)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# ──────────────── Load honest data ────────────────
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

# ──────────────── Analytical α_c lines (N→∞) ────────────────
f_ret(t)     = let φ = φ_eq(t); 1 - φ - (t/2)*log(1 - φ^2); end
α_c_gauss(t) = let fr = f_ret(t); fr >= 1 ? 0.0 : 0.5*(1 - fr)^2; end
function α_c_exact(t)
    fr = f_ret(t); fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5*log(arg)
end

# ──────────────── Finite-N V_eff boundary ────────────────
logsumexp(a, b) = let m = max(a, b); m + log(exp(a - m) + exp(b - m)); end

"""
    log_mu_per(N, βn) — log E[exp(βn·ξ·x)] for ξ uniform on N-sphere of radius √N.
"""
function log_mu_per(N::Int, βn::Float64)
    a_half = (N - 3) / 2

    # Locate the saddle of f(u) = a_half·log(1-u²) + βn·N·u on (-1, 1)
    # f'(u) = -2·a_half·u/(1-u²) + βn·N = 0
    # → βn·N·u² + (N-3)·u - βn·N = 0
    A = βn*N; B = (N - 3)
    u_max = (-B + sqrt(B^2 + 4*A^2)) / (2*A)

    f_max = a_half*log(1 - u_max^2) + βn*N*u_max
    integrand(u) = exp(a_half*log(1 - u^2) + βn*N*u - f_max)
    val, _ = quadgk(integrand, -1.0 + 1e-12, 1.0 - 1e-12; rtol=1e-10)

    log_Z = logbeta((N - 1)/2, 0.5)   # B((N-1)/2, 1/2)
    return f_max + log(val) - log_Z
end

"""
    mean_phi(α, T, N, log_mu) — ⟨φ⟩ from the 1D effective potential.
"""
function mean_phi(α::Float64, T::Float64, N::Int, log_mu::Float64; βn::Float64=1.0)
    β       = 1.0 / T
    a_half  = (N - 3) / 2
    log_Mμ  = α*N + log_mu

    # log P(m) up to constant: a_half·log(1-m²) + (β/βn)·logsumexp(βn·N·m, log_Mμ)
    log_P_un(m) = a_half*log(1 - m^2) + (β/βn)*logsumexp(βn*N*m, log_Mμ)

    # Find peak on coarse grid
    m_grid = collect(range(-0.999, 0.999, length=2001))
    log_P_max = maximum(log_P_un(m) for m in m_grid)

    Z, _   = quadgk(m -> exp(log_P_un(m) - log_P_max),    -1.0+1e-9, 1.0-1e-9; rtol=1e-8)
    num, _ = quadgk(m -> m*exp(log_P_un(m) - log_P_max),  -1.0+1e-9, 1.0-1e-9; rtol=1e-8)
    return num / Z
end

"""
    boundary_alpha(T, N) — α* where ⟨φ⟩(α*, T, N) = THR_RATIO·φ_eq(T).
"""
function boundary_alpha(t::Float64, N::Int, log_mu::Float64;
                        α_lo::Float64=0.001, α_hi::Float64=2.0, tol::Float64=1e-3)
    φ_target = THR_RATIO * φ_eq(t)
    f(α) = mean_phi(α, t, N, log_mu) - φ_target
    flo = f(α_lo); fhi = f(α_hi)
    flo * fhi > 0 && return NaN
    while (α_hi - α_lo) > tol
        αm = 0.5*(α_lo + α_hi); fm = f(αm)
        if fm * flo > 0
            α_lo = αm; flo = fm
        else
            α_hi = αm; fhi = fm
        end
    end
    return 0.5*(α_lo + α_hi)
end

println("Computing V_eff boundary curves...")
T_grid = collect(range(0.025, 0.500, length=40))
boundary = Dict{Int, Vector{Float64}}()
for N in NS_OVERLAY
    log_mu_N = log_mu_per(N, βn)
    @printf("  N=%-4d  log μ̄_per = %.4f\n", N, log_mu_N)
    boundary[N] = [boundary_alpha(t, N, log_mu_N) for t in T_grid]
end

# ──────────────── Plot ────────────────
println("Plotting overlay heatmap...")
xmin, xmax = first(alphas) - 0.01, last(alphas) + 0.01
ymin, ymax = 0.0, 0.5

p1 = heatmap(alphas, Ts, res_grid,
    color=cgrad(:RdBu), clims=(-cmax, cmax),
    xlabel="α  =  ln M / N", ylabel="T",
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    colorbar_title="⟨φ⟩ − φ_eq(T)",
    title=@sprintf("Honest LSE residual N=%d  +  V_eff boundary overlays", N_SRC),
    titlefontsize=FONT_GUIDE,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=2Plots.mm)

# Analytical N→∞ boundaries
T_range = range(max(ymin, 1e-4), ymax, length=600)
α_g = [α_c_gauss(t) for t in T_range]
mask_g = (α_g .>= xmin) .& (α_g .<= xmax)
plot!(p1, α_g[mask_g], T_range[mask_g],
      color=:blue, lw=2.0, ls=:solid, label="α_c^G(T) Gaussian (N→∞)")

α_e = [α_c_exact(t) for t in T_range]
mask_e = isfinite.(α_e) .& (α_e .>= xmin) .& (α_e .<= xmax)
plot!(p1, α_e[mask_e], T_range[mask_e],
      color=:red, lw=2.0, ls=:dash, label="α_c^E(T) Exact (N→∞)")

# Finite-N V_eff boundary curves
N_colors = Dict(25 => :purple, 100 => :darkorange, 500 => :black)
N_styles = Dict(25 => :dot, 100 => :dashdot, 500 => :solid)
for N in NS_OVERLAY
    mask = (.!isnan.(boundary[N])) .& (boundary[N] .>= xmin) .& (boundary[N] .<= xmax)
    plot!(p1, boundary[N][mask], T_grid[mask],
          color=N_colors[N], lw=1.6, ls=N_styles[N],
          label=@sprintf("V_eff  N=%d", N))
end

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir,
        @sprintf("heatmap_LSE_honest_N%d_residual_Veff_overlay.%s", N_SRC, ext)))
end
@printf("Saved: panels_paper/heatmap_LSE_honest_N%d_residual_Veff_overlay.{png,pdf}\n", N_SRC)
println("Done.")

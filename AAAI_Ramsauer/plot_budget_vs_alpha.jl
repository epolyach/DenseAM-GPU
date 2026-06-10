#=
N_SAMP budget required to observe Kramers escape at T just below the
leading-order spinodal T_sp^LO(α), as a function of α, for several N.

Free energy near the spinodal (per spin):
   ΔF(α,T)/N = (1/2) g''(φ_eq) · ξ², ξ = φ_eq(T) − (α + g_max)
   g''(φ_eq) = (1+φ_eq²)/[φ_eq(1−φ_eq²)]
Kramers escape time (units of MC steps):
   τ_esc ≈ exp(N · ΔF/T)
Detection criterion adopted here:
   T_dep = (1 − r) · T_sp^LO(α), r ∈ {0.01, 0.05, 0.10}
   N_SAMP_required = τ_esc(α, T_dep, N)

The chain must observe at least one escape event in N_SAMP steps, so we
report N_SAMP_required = max(1, τ_esc).

Caveat: at small N, thermal fluctuations in φ_1 are O(1/√N) and trip
the empirical 0.9·φ_eq departure criterion before any Kramers escape
takes place. This script does NOT model that; it isolates the Kramers
budget. The small-N curves should therefore be read as upper bounds.

Output: panels_paper/budget_NSAMP_vs_alpha.{png,pdf}
=#

using Printf
using Plots

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(φ_star) + φ_star          # ≈ 0.3774

phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))
T_sp_LO(α) = (1 - (α + g_max)^2) / (α + g_max)

function dF_per_N(α::Float64, T::Float64)
    # Exact LO barrier:  ΔF/N = (φ_eq − cusp) + (T/2) ln[(1−φ_eq²)/(1−cusp²)]
    φe = phi_eq(T)
    cusp = α + g_max
    ξ  = φe - cusp
    ξ <= 0 && return 0.0                            # basin disappeared
    return ξ + 0.5 * T * log((1 - φe^2) / (1 - cusp^2))
end

function N_SAMP_required(α::Float64, N::Int, r::Float64)
    Tsp = T_sp_LO(α)
    Tsp <= 0 && return 1.0
    Tdep = (1 - r) * Tsp
    dF   = dF_per_N(α, Tdep)
    expo = N * dF / Tdep
    return max(1.0, exp(expo))
end

# ─────────── Build curves ───────────
const ALPHAS = collect(0.20:0.005:0.615)
const NS     = [50, 100, 1000]
const R_TARGETS = [0.01, 0.05, 0.10]
const N_SAMP_CURRENT = 2000.0

# For each (r, N): N_SAMP(α)
curves = Dict{Tuple{Float64,Int}, Vector{Float64}}()
for r in R_TARGETS, N in NS
    curves[(r, N)] = [N_SAMP_required(α, N, r) for α in ALPHAS]
end

# ─────────── Plot ───────────
p = plot(size=(820, 520),
         xlabel = "α",
         ylabel = "N_SAMP required to detect escape",
         yscale = :log10,
         legend = :topright,
         framestyle = :box,
         grid = :on,
         gridalpha = 0.25,
         titlefontsize = 12,
         guidefontsize = 11,
         tickfontsize = 9,
         title = "Kramers escape budget vs α (LO formula, T_dep = (1−r)·T_sp^LO)")

# Three line styles for r
ls_for_r = Dict(0.01 => :dot, 0.05 => :solid, 0.10 => :dash)
# Three colors for N (light→dark blue)
col_for_N = Dict(50 => RGB(0.55, 0.75, 0.95),
                 100 => RGB(0.30, 0.50, 0.85),
                 1000 => RGB(0.05, 0.10, 0.45))

for r in R_TARGETS, N in NS
    label_str = "N=$N, r=$(Int(round(r*100)))%"
    plot!(p, ALPHAS, curves[(r, N)],
          lw = 1.8,
          ls = ls_for_r[r],
          color = col_for_N[N],
          label = label_str)
end

# Current MC budget line
hline!(p, [N_SAMP_CURRENT],
       lw = 2.5, ls = :solid, color = :red,
       label = "current N_SAMP = $(Int(N_SAMP_CURRENT))")

# Mark the cusp α = 1 − g_max
vline!(p, [1 - g_max],
       lw = 1.0, ls = :dashdot, color = :black, alpha = 0.5,
       label = "α = 1 − g_max ≈ 0.6226")

# Y-range
ylims!(p, (1.0, 1e12))

# Save
outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "budget_NSAMP_vs_alpha.png")
out_pdf = joinpath(outdir, "budget_NSAMP_vs_alpha.pdf")
savefig(p, out_png); savefig(p, out_pdf)

# ─────────── Console summary ───────────
println("Kramers budget summary at T_dep = 0.95·T_sp^LO(α)")
@printf("%6s | %10s | %12s | %12s | %12s\n",
        "α", "T_sp^LO", "N=50", "N=100", "N=1000")
println("-"^66)
for α in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.61, 0.615]
    Tsp = T_sp_LO(α)
    n50  = N_SAMP_required(α, 50,  0.05)
    n100 = N_SAMP_required(α, 100, 0.05)
    n1k  = N_SAMP_required(α, 1000, 0.05)
    @printf("%6.3f | %10.4f | %12.2e | %12.2e | %12.2e\n",
            α, Tsp, n50, n100, n1k)
end
println()
println("Saved:")
println("  ", out_png)
println("  ", out_pdf)

#=
Extract T_*(α) from smart-MC v18 per-disorder data — AAAI paper

Two indicators of the spurious-mode onset, both per-α scan over T:

  (1) Disorder variance of φ_1 across realisations: jumps from a thermal
      floor to a much larger value when some trials begin to leak.
  (2) Ratio ⟨φ_max_other⟩ / ⟨φ_1⟩: grows toward unity as the chain feels
      the nearest live competitor.

For each α we identify T_* as the smallest T at which an indicator crosses
its threshold. Compare to the theoretical α_c^E(T) → T_crit(α).

Usage:
  julia analyze_smart_v18_Tstar.jl 500 10000        # defaults if omitted

Output: ../panels_paper/Tstar_vs_alpha_N<N>_K<K>.{png,pdf}
        plus a table to stdout
=#

ENV["GKSwstype"] = "100"

using Plots
using Printf
using Statistics

const N_ARG = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
const K_ARG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000

const FIG_DPI    = 300
const FIG_W      = round(Int, 86/25.4*100) + 60
const FIG_H      = round(Int, 86/25.4*100)
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_LEG   = 6
default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK, legendfontsize=FONT_LEG)

# ──────────────── Read CSV ────────────────
csv_in = joinpath(@__DIR__,
    @sprintf("basin_stab_LSE_smart_v18_N%d_K%d.csv", N_ARG, K_ARG))
isfile(csv_in) || error("Missing $csv_in")
println("Reading $csv_in ...")
lines = filter(l -> !startswith(l, "#") && !isempty(l), readlines(csv_in))[2:end]
n = length(lines)
alpha = zeros(n); T = zeros(n); phi_a = zeros(n); phi_b = zeros(n); phimax = zeros(n)
for i in 1:n
    f = split(lines[i], ",")
    alpha[i]  = parse(Float64, f[1]); T[i]    = parse(Float64, f[2])
    phi_a[i]  = parse(Float64, f[4]); phi_b[i] = parse(Float64, f[5])
    phimax[i] = parse(Float64, f[7])
end
αs = sort(unique(round.(alpha, digits=4)))
@printf("Rows: %d   α values: %d   α∈[%.3f, %.3f]\n",
        n, length(αs), first(αs), last(αs))

# ──────────────── Theory: T_crit(α) and φ_eq(T) ────────────────
φ_eq(T) = 0.5*(-T + sqrt(T^2 + 4))
f_ret(T) = let φ = φ_eq(T); 1 - φ - (T/2)*log(1 - φ^2); end
function αcE(T)
    fr = f_ret(T); fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5*log(arg)
end
function T_crit_theory(α; lo=1e-4, hi=2.0)
    for _ in 1:80
        m = (lo+hi)/2; αcE(m) > α ? (lo = m) : (hi = m)
    end
    (lo+hi)/2
end

# ──────────────── Per-(α, T) statistics with bootstrap ────────────────
function bootstrap_stat(vals::Vector{Float64}, statfn::Function; nB::Int=200)
    isempty(vals) && return (NaN, NaN)
    s0 = statfn(vals)
    n = length(vals)
    bs = zeros(nB)
    for b in 1:nB
        idx = rand(1:n, n)
        bs[b] = statfn(vals[idx])
    end
    return (s0, std(bs))
end

# For each α: per-T arrays  Var_d(T), ⟨φ_max_other⟩(T), ⟨φ_1⟩(T)
struct ColumnStats
    Ts::Vector{Float64}
    varφ::Vector{Float64}    ;  varφ_err::Vector{Float64}
    pmax::Vector{Float64}    ;  pmax_err::Vector{Float64}
    pmean::Vector{Float64}
    surv::Vector{Float64}                 # survival probability: fraction of disorder samples with φ > 0.5
end

function per_α_stats(α)
    mask_α = abs.(alpha .- α) .< 1e-4
    Ts_h = sort(unique(round.(T[mask_α], digits=6)))
    varφ = Float64[]; varφ_err = Float64[]
    pmax = Float64[]; pmax_err = Float64[]
    pmean = Float64[]; surv = Float64[]
    for t in Ts_h
        mask = mask_α .& (abs.(T .- t) .< 1e-6)
        φvals = vcat(phi_a[mask], phi_b[mask])
        pmvals = phimax[mask]
        push!(pmean, mean(φvals))
        push!(surv, mean(φvals .> 0.5))
        v, ve = bootstrap_stat(φvals, var)
        push!(varφ, v); push!(varφ_err, ve)
        p, pe = bootstrap_stat(pmvals, mean)
        push!(pmax, p); push!(pmax_err, pe)
    end
    return ColumnStats(Ts_h, varφ, varφ_err, pmax, pmax_err, pmean, surv)
end

# T_*(1):  T at which log(Var_d) crosses the geometric mean of its (low, high)
#          extremes — i.e., the mid-rise on a log scale.
function find_Tstar_var(c::ColumnStats)
    nT = length(c.Ts); nT < 5 && return NaN
    nf = min(3, nT)
    floor_v = median(c.varφ[1:nf])
    floor_v <= 0 && (floor_v = minimum(c.varφ[c.varφ .> 0]; init=1e-12))
    high_v = maximum(c.varφ)
    high_v <= floor_v && return NaN
    target = sqrt(floor_v * high_v)        # mid-rise on log scale
    for i in 2:nT
        if c.varφ[i] > target
            # linear interp in log
            x0, x1 = log(c.varφ[i-1]), log(c.varφ[i])
            t0, t1 = c.Ts[i-1], c.Ts[i]
            return t0 + (t1-t0)*(log(target) - x0)/(x1 - x0)
        end
    end
    return NaN
end

# T_*(surv):  T at which the survival probability (fraction of disorder samples
#             with φ > 0.5) drops below a threshold (default 0.5 = half escape).
function find_Tstar_survival(c::ColumnStats; thresh::Float64=0.5)
    nT = length(c.Ts); nT < 5 && return NaN
    for i in 2:nT
        if c.surv[i] < thresh
            x0, x1 = c.surv[i-1], c.surv[i]
            t0, t1 = c.Ts[i-1], c.Ts[i]
            return t0 + (t1-t0)*(thresh - x0)/(x1 - x0)
        end
    end
    return NaN
end

# ──────────────── Sweep α ────────────────
Tstar1 = Float64[]; TstarS = Float64[]; Tcrit_th = Float64[]
println()
@printf("%-6s  %-10s  %-10s  %-10s  %-8s  %-8s\n",
        "α", "T_crit_th", "T_*(var)", "T_*(surv)", "rel(1)", "rel(S)")
println("─"^70)
for α in αs
    c = per_α_stats(α)
    t1 = find_Tstar_var(c)
    tS = find_Tstar_survival(c)
    tcth = T_crit_theory(α)
    push!(Tstar1, t1); push!(TstarS, tS); push!(Tcrit_th, tcth)
    @printf("%.2f   %-10.4f  %-10.4f  %-10.4f  %-8.2f  %-8.2f\n",
            α, tcth, t1, tS,
            isnan(t1) ? NaN : t1/tcth, isnan(tS) ? NaN : tS/tcth)
end

# ──────────────── Plot T_*(α) vs α_c^E(T) ────────────────
out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

p = plot(xlabel="α", ylabel="T",
         title=@sprintf("Smart-MC T_*(α) vs theoretical α_c^E(T)\nN=%d  K=%d", N_ARG, K_ARG),
         titlefontsize=FONT_GUIDE,
         xlims=(first(αs)-0.02, last(αs)+0.02),
         ylims=(0.005, 1.0), yscale=:log10,
         legend=:topright,
         size=(FIG_W, FIG_H), dpi=FIG_DPI,
         left_margin=4Plots.mm, bottom_margin=4Plots.mm, top_margin=2Plots.mm)

# Theory curve
T_range = 10 .^ range(log10(0.005), log10(1.0), length=400)
α_theory = [αcE(t) for t in T_range]
mask = isfinite.(α_theory) .& (α_theory .>= first(αs)) .& (α_theory .<= last(αs))
plot!(p, α_theory[mask], T_range[mask],
      color=:black, lw=2.5, ls=:dash, label="α_c^E(T)  theory")

# T_*^(1) — variance indicator
mask1 = isfinite.(Tstar1)
plot!(p, αs[mask1], Tstar1[mask1], lw=1.5, marker=:circle, ms=4,
      color=:firebrick, label="T_*  from Var_d(φ_1)")

# T_*(surv) — survival probability indicator
maskS = isfinite.(TstarS)
plot!(p, αs[maskS], TstarS[maskS], lw=1.5, marker=:utriangle, ms=4,
      color=:royalblue, label="T_*  from survival (50% escape)")

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir,
        @sprintf("Tstar_vs_alpha_N%d_K%d.%s", N_ARG, K_ARG, ext)))
end
@printf("Saved: panels_paper/Tstar_vs_alpha_N%d_K%d.{png,pdf}\n",
        N_ARG, K_ARG)

# ──────────────── Diagnostic: also plot the indicators vs T at a few α ────────────────
α_diag = [0.40, 0.50, 0.70, 1.00]
p2 = plot(layout=(2, length(α_diag)),
          size=(FIG_W*length(α_diag), FIG_H*2), dpi=FIG_DPI,
          left_margin=4Plots.mm, bottom_margin=4Plots.mm, top_margin=2Plots.mm)
for (j, α) in enumerate(α_diag)
    c = per_α_stats(α)
    tcth = T_crit_theory(α)
    # top row: Var_d(T)
    plot!(p2[1, j], c.Ts, c.varφ, yerror=c.varφ_err,
          marker=:circle, ms=3, lw=1, color=:firebrick,
          xlabel="T", ylabel="Var_d(φ_1)",
          title=@sprintf("α=%.2f  T_crit_th=%.3f", α, tcth),
          titlefontsize=FONT_GUIDE,
          yscale=:log10, legend=false)
    vline!(p2[1, j], [tcth], color=:black, ls=:dash, lw=1.5)
    # bottom row: survival probability(T)
    plot!(p2[2, j], c.Ts, c.surv,
          marker=:utriangle, ms=3, lw=1, color=:royalblue,
          xlabel="T", ylabel="P(φ > 0.5)  survival",
          ylims=(-0.02, 1.05), legend=false)
    vline!(p2[2, j], [tcth], color=:black, ls=:dash, lw=1.5)
    hline!(p2[2, j], [0.5], color=:gray, ls=:dot, lw=0.8)
end
for ext in ("png", "pdf")
    savefig(p2, joinpath(out_dir,
        @sprintf("Tstar_diagnostics_N%d_K%d.%s", N_ARG, K_ARG, ext)))
end
@printf("Saved: panels_paper/Tstar_diagnostics_N%d_K%d.{png,pdf}\n",
        N_ARG, K_ARG)

println("\nDone.")

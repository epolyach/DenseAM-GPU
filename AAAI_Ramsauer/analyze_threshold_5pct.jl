#=
Physically motivated data boundary threshold.

Current paper boundary uses Z = <phi> - phi_eq(T) and flags a cell as broken
when Z < -0.01. This is a bulk-average proxy. A cleaner physical criterion:
flag the cell as broken when the fraction of chain endpoints sitting OUTSIDE
the retrieval basin first exceeds f* (e.g. 5%).

Relation:
  Z = f_esc * (phi_SV - phi_eq(T)) = -f_esc * Delta_phi(alpha, T)
so for a fixed f_esc,
  |Z|_threshold = f_esc * Delta_phi(alpha, T),
which is NOT a constant in (alpha, T).

This script:
  1) reads the truncated MC CSV at M = 4.4e6,
  2) at each (alpha, T) counts the empirical escape fraction via a fixed cut
     phi < phi_eq(T) - GAP (GAP large enough to be outside intra-basin spread,
     small enough to catch real escapers),
  3) reports, at a few representative T slices, how Z = -0.01 translates to
     f_esc across alpha, and what |Z| corresponds to f_esc = 0.05,
  4) extracts two data boundaries from the same dataset:
       a) Z < -0.01  (current paper),
       b) f_esc > 0.05  (proposed),
     and saves a comparison heatmap.

Output: panels_paper/threshold_5pct_comparison.{png,pdf}
=#

using Printf
using Plots
using Statistics
using LaTeXStrings

default(dpi = 200)

const phi_star = (sqrt(5) - 1) / 2
const g_max    = 0.5 * log(1 - phi_star^2) + phi_star
phi_eq(T)      = 0.5 * (-T + sqrt(T^2 + 4))

const CSV_IN = joinpath(@__DIR__,
    "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e+06_phikeep0.40.csv")

# Cut for "escaped" classification: chain endpoint phi < phi_eq(T) - GAP.
# Intra-basin spread is dominated by 1/sqrt(N) ~ 0.01-0.02 here; GAP = 0.15
# is comfortably outside that, and well above where escaped chains land
# (typically phi ~ 0.4-0.65 at T ~ 0.3, vs phi_eq ~ 0.86).
const GAP = 0.15

# ---- Read all per-chain phi values per (alpha, T) cell --------------------
cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
open(CSV_IN, "r") do f
    for line in eachline(f)
        startswith(line, "#") && continue
        startswith(line, "alpha") && continue
        fs = split(line, ",")
        α  = parse(Float64, fs[1])
        T  = parse(Float64, fs[2])
        φa = parse(Float64, fs[6])
        φb = parse(Float64, fs[7])
        key = (α, T)
        haskey(cells, key) || (cells[key] = Float64[])
        push!(cells[key], φa, φb)
    end
end

αs = sort(unique(k[1] for k in keys(cells)))
Ts = sort(unique(k[2] for k in keys(cells)))
@printf("α grid: %d  [%.3f .. %.3f]\n", length(αs), first(αs), last(αs))
@printf("T grid: %d  [%.3f .. %.3f]\n", length(Ts), first(Ts), last(Ts))

# ---- Per-cell quantities --------------------------------------------------
Zmat      = fill(NaN, length(Ts), length(αs))
Fmat      = fill(NaN, length(Ts), length(αs))   # empirical f_esc
PhiSVmat  = fill(NaN, length(Ts), length(αs))   # empirical mean phi of escaped
DeltaPhi  = fill(NaN, length(Ts), length(αs))   # phi_eq - phi_SV (empirical)
Nchainmat = zeros(Int, length(Ts), length(αs))

for (iα, α) in enumerate(αs), (iT, T) in enumerate(Ts)
    haskey(cells, (α, T)) || continue
    φs = cells[(α, T)]
    isempty(φs) && continue
    Nchainmat[iT, iα] = length(φs)
    Zmat[iT, iα]      = mean(φs) - phi_eq(T)
    cut               = phi_eq(T) - GAP
    escaped           = φs .< cut
    nesc              = count(escaped)
    Fmat[iT, iα]      = nesc / length(φs)
    if nesc > 0
        PhiSVmat[iT, iα] = mean(@view φs[escaped])
        DeltaPhi[iT, iα] = phi_eq(T) - PhiSVmat[iT, iα]
    end
end

# ---- Diagnostic table at a few T slices ----------------------------------
println()
println("Per-cell diagnostic: how Z translates to f_esc.")
println("Columns: α,  Z,  f_esc(empirical),  Δφ(empirical),  TH_5pct=0.05·Δφ")
for T in (0.005, 0.105, 0.205, 0.305, 0.405)
    iT = findfirst(t -> isapprox(t, T; atol=1e-6), Ts)
    iT === nothing && continue
    @printf("\nT = %.3f  (phi_eq = %.4f)\n", T, phi_eq(T))
    @printf("  %-7s %10s %12s %12s %12s\n", "α", "Z", "f_esc", "Δφ_emp", "TH(5%)")
    for iα in eachindex(αs)
        Z = Zmat[iT, iα]; f = Fmat[iT, iα]
        isnan(Z) && continue
        # Show only cells with non-trivial escape or near boundary
        (abs(Z) < 1e-3 && f < 1e-3) && continue
        dp = DeltaPhi[iT, iα]
        th = isnan(dp) ? NaN : 0.05 * dp
        @printf("  %-7.3f %10.4f %12.4f %12.4f %12.4f\n",
                αs[iα], Z, f, dp, th)
    end
end

# ---- Extract two boundaries (raw, no smoothing) ---------------------------
function boundary_locus(αs, Ts, M, predicate)
    α_bd = Float64[]
    T_bd = Float64[]
    for (iα, α) in enumerate(αs)
        for (iT, T) in enumerate(Ts)
            v = M[iT, iα]
            (isnan(v) || !predicate(v)) && continue
            push!(α_bd, α)
            push!(T_bd, T)
            break
        end
    end
    return α_bd, T_bd
end

αZ, TZ = boundary_locus(αs, Ts, Zmat, z -> z < -0.01)
αF, TF = boundary_locus(αs, Ts, Fmat, f -> f > 0.05)

@printf("\nBoundary (Z < -0.01):     %d α columns  [%.3f .. %.3f]\n",
        length(αZ), isempty(αZ) ? NaN : first(αZ), isempty(αZ) ? NaN : last(αZ))
@printf("Boundary (f_esc > 0.05):  %d α columns  [%.3f .. %.3f]\n",
        length(αF), isempty(αF) ? NaN : first(αF), isempty(αF) ? NaN : last(αF))

# ---- ΔT between boundaries at common α -----------------------------------
println("\nΔT(5%-Z) = T_bd(f_esc>0.05) - T_bd(Z<-0.01)  at common α:")
mZ = Dict(round(α, digits=4) => T for (α, T) in zip(αZ, TZ))
mF = Dict(round(α, digits=4) => T for (α, T) in zip(αF, TF))
common = sort(intersect(collect(keys(mZ)), collect(keys(mF))))
for α in common
    @printf("  α=%.3f   T_Z=%.4f   T_F=%.4f   ΔT=%+0.4f\n",
            α, mZ[α], mF[α], mF[α] - mZ[α])
end

# ---- Plots ---------------------------------------------------------------
# (1) heatmap of f_esc with both boundaries overlaid
p1 = heatmap(αs, Ts, Fmat;
             xlabel = L"\alpha", ylabel = L"T",
             c = :viridis, clims = (0.0, 1.0),
             colorbar_title = L"f_{\rm esc}",
             framestyle = :box, tick_direction = :out,
             title = L"f_{\rm esc}(\alpha, T) ~~M=4.4\times10^6",
             titlefontsize = 10, guidefontsize = 10, tickfontsize = 8,
             xlims = (0.20, 0.50), ylims = (0.0, 0.5),
             legendfontsize = 7,
             size = (700, 480))
plot!(p1, αZ, TZ; color = :red,    lw = 2.2, label = "Z < -0.01")
plot!(p1, αF, TF; color = :white,  lw = 2.2, ls = :dash,
      label = "f_esc > 0.05")

# (2) heatmap of |Z|/Δφ implied escape fraction (analytic interpretation)
#     using analytic Δφ = phi_eq(T) - (alpha + g_max) where positive.
ImpliedF = fill(NaN, length(Ts), length(αs))
for (iα, α) in enumerate(αs), (iT, T) in enumerate(Ts)
    dp = phi_eq(T) - (α + g_max)
    dp > 0 || continue
    Z = Zmat[iT, iα]
    isnan(Z) && continue
    ImpliedF[iT, iα] = -Z / dp     # since Z is negative inside broken phase
end

p2 = heatmap(αs, Ts, ImpliedF;
             xlabel = L"\alpha", ylabel = L"T",
             c = :viridis, clims = (0.0, 1.0),
             colorbar_title = L"|Z|\,/\,\Delta\varphi",
             framestyle = :box, tick_direction = :out,
             title = L"\rm implied~ f_{\rm esc}~from~Z/\Delta\varphi",
             titlefontsize = 10, guidefontsize = 10, tickfontsize = 8,
             xlims = (0.20, 0.50), ylims = (0.0, 0.5),
             legendfontsize = 7,
             size = (700, 480))
plot!(p2, αZ, TZ; color = :red,    lw = 2.2, label = "Z < -0.01")
plot!(p2, αF, TF; color = :white,  lw = 2.2, ls = :dash,
      label = "f_esc > 0.05")

P = plot(p1, p2; layout = (1, 2), size = (1280, 480),
         left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "threshold_5pct_comparison.png")
out_pdf = joinpath(outdir, "threshold_5pct_comparison.pdf")
savefig(P, out_png); savefig(P, out_pdf)
println("\nSaved: $out_png")

# ---- Field of TH(α, T) for fixed f_esc = 0.05 (analytic) ------------------
println("\nAnalytic TH = 0.05 · (phi_eq(T) - (alpha + g_max)) at representative cells:")
for T in (0.0, 0.1, 0.2, 0.3, 0.4)
    println()
    @printf("T = %.2f  (phi_eq = %.4f)\n", T, phi_eq(T))
    @printf("  %-7s %12s %12s\n", "α", "Δφ_analytic", "TH(5%)")
    for α in (0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.40, 0.43)
        dp = phi_eq(T) - (α + g_max)
        dp > 0 || (println(@sprintf("  %-7.3f  -- past spinodal --", α)); continue)
        @printf("  %-7.3f %12.4f %12.4f\n", α, dp, 0.05 * dp)
    end
end

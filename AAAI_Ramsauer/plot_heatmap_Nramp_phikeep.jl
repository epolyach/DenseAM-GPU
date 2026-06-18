#=
Heatmap of <phi> − phi_eq(T) for the semismart Nramp run with PHI_KEEP fixed
(basin_stab_LSE_semismart_AAAI_Nramp_M4.4e+06_phikeep0.40.csv).

No inset. Four analytical boundary curves overlaid:
  blue solid          Gaussian baseline      ac_gauss(T)
  red solid           saddle-dominated       ac_sd(T)
  orange dashed       exact-density edge     ac_bd(T)
  purple dotted       spinodal               ac_sp(T)

Optional CLI:
  --boundary=<th>  outline the empirical data boundary as the locus
                   where Z(α, T) = ⟨φ⟩ − φ_eq(T) first drops below −th
                   (scanning α upward at each T). Drawn as a thick black line.

Output: panels_paper/heatmap_Nramp_M4.4e6_phikeep0.40[_bd<th>].{png,pdf}
=#

using Printf
using Plots
using Statistics

function parse_boundary_threshold(args)
    for a in args
        startswith(a, "--boundary=") || continue
        v = split(a, "=", limit=2)[2]
        try
            return parse(Float64, v)
        catch
            @warn "could not parse $a as Float64; ignoring"
            return nothing
        end
    end
    return nothing
end

const BOUNDARY_TH = parse_boundary_threshold(ARGS)

const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(1 - φ_star^2) + φ_star          # ≈ 0.3774

phi_eq(T) = 0.5*(-T + sqrt(T^2 + 4))
fret(T)   = (1 - phi_eq(T)) - 0.5*T*log(1 - phi_eq(T)^2)

ac_gauss(T) = 0.5 * (1 - fret(T))^2
ac_bd(T)    = -0.5 * log(1 - (1 - fret(T))^2)
ac_sd(T)    = 1 - g_max - fret(T)
ac_sp(T)    = phi_eq(T) - g_max

# ─────────── Read CSV ───────────
const CSV_IN = joinpath(@__DIR__,
    "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e+06_phikeep0.40.csv")

cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
Ns    = Dict{Tuple{Float64,Float64}, Int}()

open(CSV_IN, "r") do f
    for line in eachline(f)
        startswith(line, "#") && continue
        startswith(line, "alpha") && continue
        fs = split(line, ",")
        α     = parse(Float64, fs[1])
        T     = parse(Float64, fs[2])
        Nused = parse(Int,     fs[3])
        φa    = parse(Float64, fs[6])
        φb    = parse(Float64, fs[7])
        key = (α, T)
        if !haskey(cells, key)
            cells[key] = Float64[]
            Ns[key]    = Nused
        end
        push!(cells[key], φa, φb)
    end
end

αs = sort(unique(k[1] for k in keys(cells)))
Ts = sort(unique(k[2] for k in keys(cells)))
@printf("α grid: %d points from %.3f to %.3f\n", length(αs), first(αs), last(αs))
@printf("T grid: %d points from %.3f to %.3f\n", length(Ts), first(Ts), last(Ts))

Z = fill(NaN, length(Ts), length(αs))
for (iα, α) in enumerate(αs), (iT, T) in enumerate(Ts)
    haskey(cells, (α, T)) || continue
    isempty(cells[(α, T)]) && continue
    Z[iT, iα] = mean(cells[(α, T)]) - phi_eq(T)
end

# ─────────── Plot ───────────
p = heatmap(αs, Ts, Z,
            xlabel = "α",
            ylabel = "T",
            title  = "⟨φ⟩ − φ_eq(T), semismart Nramp (M=4.4e6, φ_keep=0.40)",
            c      = :RdBu,
            clims  = (-0.9, 0.9),
            colorbar_title = "⟨φ⟩ − φ_eq(T)",
            size   = (900, 540),
            framestyle = :box,
            titlefontsize = 11, guidefontsize = 11, tickfontsize = 9)

T_curve = collect(0.005:0.005:0.495)
plot!(p, [ac_gauss(t) for t in T_curve], T_curve,
      color = :blue,   lw = 2.0, ls = :solid, label = "Gauss")
plot!(p, [ac_sd(t)    for t in T_curve], T_curve,
      color = :red,    lw = 2.0, ls = :solid, label = "saddle")
plot!(p, [ac_bd(t)    for t in T_curve], T_curve,
      color = :orange, lw = 2.0, ls = :dash,  label = "exact")
plot!(p, [ac_sp(t)    for t in T_curve], T_curve,
      color = :purple, lw = 1.8, ls = :dot,   label = "spinodal")

# Data boundary as a staircase ON cell edges: for each α column, lowest T at
# which Z drops below −BOUNDARY_TH (scan T upward). The boundary outlines the
# basin: horizontal segments at the BOTTOM edge of the first-broken cell
# (T = T_bd − dT/2), vertical segments at the cell edge between consecutive
# α columns (α = α_bd + dα/2).
if BOUNDARY_TH !== nothing
    α_bd = Float64[]
    T_bd = Float64[]
    for (iα, α) in enumerate(αs)
        for (iT, T) in enumerate(Ts)
            if !isnan(Z[iT, iα]) && Z[iT, iα] < -BOUNDARY_TH
                push!(α_bd, α)
                push!(T_bd, T)
                break
            end
        end
    end
    if !isempty(α_bd)
        dα = length(αs) > 1 ? αs[2] - αs[1] : 0.0
        dT = length(Ts) > 1 ? Ts[2] - Ts[1] : 0.0
        # Build edge-aligned polyline. For column i: horizontal at T_i - dT/2
        # from α_i - dα/2 to α_i + dα/2, then vertical at α_i + dα/2 to
        # T_{i+1} - dT/2 (if there is a next column).
        αp = Float64[]
        Tp = Float64[]
        for i in eachindex(α_bd)
            αl = α_bd[i] - dα/2
            αr = α_bd[i] + dα/2
            Tb = T_bd[i] - dT/2
            if i == 1
                push!(αp, αl); push!(Tp, Tb)
            end
            push!(αp, αr); push!(Tp, Tb)
            if i < length(α_bd)
                Tb_next = T_bd[i+1] - dT/2
                push!(αp, αr); push!(Tp, Tb_next)
            end
        end
        plot!(p, αp, Tp;
              color = :black, lw = 2.5, ls = :solid,
              label = @sprintf("data (Z < −%.3g)", BOUNDARY_TH))
        @printf("Data boundary plotted at threshold −%.4f (%d / %d α columns)\n",
                BOUNDARY_TH, length(α_bd), length(αs))
    else
        @warn @sprintf("No α column had Z < −%.4f; data boundary skipped", BOUNDARY_TH)
    end
end

xlims!(p, (first(αs), last(αs)))
ylims!(p, (0.0, 0.5))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
suffix  = BOUNDARY_TH === nothing ? "" : @sprintf("_bd%.3g", BOUNDARY_TH)
out_png = joinpath(outdir, "heatmap_Nramp_M4.4e6_phikeep0.40$(suffix).png")
out_pdf = joinpath(outdir, "heatmap_Nramp_M4.4e6_phikeep0.40$(suffix).pdf")
savefig(p, out_png); savefig(p, out_pdf)
println("Saved:")
println("  ", out_png)
println("  ", out_pdf)

N_min = minimum(values(Ns))
N_max = maximum(values(Ns))
@printf("N range used in this dataset: %d … %d\n", N_min, N_max)

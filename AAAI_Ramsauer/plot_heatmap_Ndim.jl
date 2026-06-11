#=
Joint heatmap: overlay the N-dim hybrid MC insert on the
honest+truncated background.  Hybrid CSV: hybrid_Ndim_LSE_N100.csv.

Output: panels_paper/heatmap_honest_over_Nramp_N100_Ndim.{png,pdf,eps}
=#

using Printf
using Plots
using Statistics

const φ_star = (sqrt(5)-1)/2
const g_max  = 0.5*log(φ_star) + φ_star

phi_eq(T)   = 0.5 * (-T + sqrt(T^2 + 4))
fret(T)     = (1 - phi_eq(T)) - 0.5 * T * log(1 - phi_eq(T)^2)
ac_gauss(T) = 0.5 * (1 - fret(T))^2
ac_bd(T)    = -0.5 * log(1 - (1 - fret(T))^2)
ac_sd(T)    = 1 - g_max - fret(T)
ac_sp(T)    = phi_eq(T) - g_max

function load_phi_mean(path; phi_cols::Vector{Int})
    cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
    isfile(path) || return cells
    open(path, "r") do f
        for line in eachline(f)
            (isempty(line) || startswith(line, "#") || startswith(line, "alpha") || startswith(line, "generator")) && continue
            fs = split(line, ",")
            length(fs) <= maximum(phi_cols) && continue
            α = parse(Float64, fs[1])
            T = parse(Float64, fs[2])
            key = (round(α, digits=4), round(T, digits=5))
            haskey(cells, key) || (cells[key] = Float64[])
            for j in phi_cols
                push!(cells[key], parse(Float64, fs[j]))
            end
        end
    end
    return cells
end

const SEMI_CSV = joinpath(@__DIR__, "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6.csv")
const HYB_CSV  = joinpath(@__DIR__, "hybrid_Ndim_LSE_N100.csv")

cells_semi = load_phi_mean(SEMI_CSV; phi_cols=[6, 7])
cells_hyb  = load_phi_mean(HYB_CSV;  phi_cols=[5])

αs_base = sort(unique([k[1] for k in keys(cells_semi)]))
Ts_base = sort(unique([k[2] for k in keys(cells_semi)]))

αs_hyb_grid = collect(0.50:0.01:0.62)
Ts_hyb_grid = collect(0.0025:0.005:0.0975)
αs_all = sort(unique(vcat(αs_base, αs_hyb_grid)))
Ts_fine = collect(0.0025:0.005:0.4975)

const HYB_RECT = (αs_hyb_grid[1] - 0.005, αs_hyb_grid[end] + 0.005,
                  0.0, Ts_hyb_grid[end] + 0.0025)

function column_points(α)
    αk = round(α, digits=4)
    pts = Tuple{Float64,Float64}[]
    used = Set{Float64}()
    for T in Ts_hyb_grid
        key = (αk, round(T, digits=5))
        if haskey(cells_hyb, key) && !isempty(cells_hyb[key])
            push!(pts, (T, mean(cells_hyb[key])))
            push!(used, round(T, digits=5))
        end
    end
    for T in Ts_base
        Tk = round(T, digits=5)
        Tk in used && continue
        key = (αk, Tk)
        v = haskey(cells_semi, key) && !isempty(cells_semi[key]) ?
                mean(cells_semi[key]) : NaN
        if !isnan(v)
            push!(pts, (T, v))
            push!(used, Tk)
        end
    end
    sort!(pts; by = first)
    return pts
end

function lin_interp(points, T)
    isempty(points) && return NaN
    n = length(points)
    points[1][1] >= T && return points[1][2]
    points[n][1] <= T && return points[n][2]
    for i in 2:n
        if points[i][1] >= T
            t1, v1 = points[i-1]; t2, v2 = points[i]
            return v1 + (v2 - v1)*(T - t1)/(t2 - t1)
        end
    end
    return points[n][2]
end

Z = fill(NaN, length(Ts_fine), length(αs_all))
for (iα, α) in enumerate(αs_all)
    pts = column_points(α)
    isempty(pts) && continue
    for (iT, T) in enumerate(Ts_fine)
        v = lin_interp(pts, T)
        isnan(v) || (Z[iT, iα] = v - phi_eq(T))
    end
end

p = heatmap(αs_all, Ts_fine, Z;
            xlabel = "α", ylabel = "T",
            c = :RdBu, clims = (-0.9, 0.9),
            colorbar_title = "⟨φ⟩ − φ_eq(T)",
            size = (940, 540), framestyle = :box,
            guidefontsize = 11, tickfontsize = 9)
T_curve = collect(0.005:0.005:0.495)
plot!(p, [ac_gauss(t) for t in T_curve], T_curve, color=:blue,   lw=2.0, ls=:solid, label="Gauss")
plot!(p, [ac_bd(t)    for t in T_curve], T_curve, color=:orange, lw=2.0, ls=:dash,  label="exact")
plot!(p, [ac_sd(t)    for t in T_curve], T_curve, color=:red,    lw=2.0, ls=:solid, label="saddle")
plot!(p, [ac_sp(t)    for t in T_curve], T_curve, color=:purple, lw=1.8, ls=:dot,   label="spinodal")

x1, x2, y1, y2 = HYB_RECT
plot!(p, [x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1],
      color=RGBA(0.5,0.5,0.5,0.9), lw=1.5, ls=:solid, label="hybrid N-dim insert")
xlims!(p, (0.20, 0.70)); ylims!(p, (0.0, 0.5))

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
stem = "heatmap_honest_over_Nramp_N100_Ndim"
out_png = joinpath(outdir, stem*".png")
out_pdf = joinpath(outdir, stem*".pdf")
out_eps = joinpath(outdir, stem*".eps")
savefig(p, out_png); savefig(p, out_pdf)
run(`pdftops -eps $out_pdf $out_eps`)
println("Saved: ", out_png, "  ", out_pdf, "  ", out_eps)

#=
Analyse spinodal_probe_cpu.csv: per (α, N), find T at which cold-chain
residual ⟨φ⟩ − φ_eq(T) drops below a threshold (single-pattern basin loss).
The cold-chain boundary at the largest N is the empirical spinodal T_sp(α).

Also produces a diagnostic plot showing cold vs hot chains.
=#

using Printf, Statistics, Plots

const F = Float32

# Re-define φ_eq for residual computation
phi_eq(t::Float64) = 0.5 * (-t + sqrt(t^2 + 4))

# Read CSV
function load_csv(path::String)
    lines = filter(l -> !startswith(l, "#") && !isempty(l), readlines(path))
    lines = lines[2:end]
    rows = Tuple{Float64, Int, Float64, Int, Symbol, Float64}[]
    for l in lines
        f = split(l, ",")
        α    = parse(Float64, f[1])
        N    = parse(Int,     f[2])
        T    = parse(Float64, f[3])
        d    = parse(Int,     f[4])
        init = Symbol(f[5])
        phi  = parse(Float64, f[6])
        push!(rows, (α, N, T, d, init, phi))
    end
    return rows
end

# Group rows by (α, N, init), averaging over disorder
function group_by_aNi(rows)
    by_aNi = Dict{Tuple{Float64,Int,Symbol}, Vector{Tuple{Float64,Float64}}}()
    for (α, N, T, d, init, phi) in rows
        key = (α, N, init)
        push!(get!(by_aNi, key, []), (T, phi))
    end
    # Average over disorder per T
    result = Dict{Tuple{Float64,Int,Symbol}, Vector{Tuple{Float64,Float64}}}()
    for (key, vec) in by_aNi
        by_T = Dict{Float64, Vector{Float64}}()
        for (T, phi) in vec
            push!(get!(by_T, T, Float64[]), phi)
        end
        result[key] = sort([(T, mean(phis)) for (T, phis) in by_T])
    end
    return result
end

# Find T at which the cold chain first departs from φ_eq(T) — i.e.
# φ falls below a tight fraction of φ_eq. "0.90·φ_eq" captures the
# leading edge of basin loss rather than the half-way point.
function find_T_boundary(curve::Vector{Tuple{Float64,Float64}}, thresh::Float64=0.90)
    for i in 1:length(curve)-1
        T, phi = curve[i]
        T2, phi2 = curve[i+1]
        r1 = phi / phi_eq(T)
        r2 = phi2 / phi_eq(T2)
        if r1 > thresh && r2 < thresh
            return T + (T2 - T) * (r1 - thresh) / (r1 - r2)
        end
    end
    return NaN
end

# Main analysis
rows = load_csv("spinodal_probe_cpu.csv")
@printf("Loaded %d rows\n", length(rows))

g = group_by_aNi(rows)
αs  = sort(unique([k[1] for k in keys(g)]))
Ns  = sort(unique([k[2] for k in keys(g)]))

# Summary table: T_sp(α, N, init)
println()
println("Cold-chain boundary (T where ⟨φ⟩ drops below 0.5·φ_eq(T)):")
@printf("  %s  %s  %s\n", lpad("α", 6),
        join([lpad("N=$N", 8) for N in Ns], "  "),
        lpad("T_sp_LO", 10))
const g_max_paper = 0.5*log((sqrt(5)-1)/2) + (sqrt(5)-1)/2
T_sp_LO(α) = (1 - (α + g_max_paper)^2) / (α + g_max_paper)
for α in αs
    cells = String[]
    for N in Ns
        c = get(g, (α, N, :cold), nothing)
        T_app = c === nothing ? NaN : find_T_boundary(c)
        push!(cells, @sprintf("%8.4f", T_app))
    end
    @printf("  %6.3f  %s  %10.4f\n", α, join(cells, "  "), T_sp_LO(α))
end

println()
println("Hot-chain boundary (where hot residual stays ≥ 0.5·φ_eq):")
@printf("  %s  %s\n", lpad("α", 6),
        join([lpad("N=$N", 8) for N in Ns], "  "))
for α in αs
    cells = String[]
    for N in Ns
        c = get(g, (α, N, :hot), nothing)
        T_app = c === nothing ? NaN : find_T_boundary(c)
        push!(cells, @sprintf("%8.4f", T_app))
    end
    @printf("  %6.3f  %s\n", α, join(cells, "  "))
end

# Save a digest CSV
open("spinodal_summary.csv", "w") do f
    write(f, "alpha,N,T_sp_cold,T_sp_hot,T_sp_LO\n")
    for α in αs, N in Ns
        cold = get(g, (α, N, :cold), nothing)
        hot  = get(g, (α, N, :hot),  nothing)
        T_cold = cold === nothing ? NaN : find_T_boundary(cold)
        T_hot  = hot  === nothing ? NaN : find_T_boundary(hot)
        @printf(f, "%.3f,%d,%.4f,%.4f,%.4f\n", α, N, T_cold, T_hot, T_sp_LO(α))
    end
end
println("\nSaved digest: spinodal_summary.csv")

# Plot: cold and hot residuals vs T, per α, with N as color
ENV["GKSwstype"] = "100"
default(guidefontsize=8, tickfontsize=7, legendfontsize=6)
out_dir = joinpath(@__DIR__, "panels_paper")
mkpath(out_dir)

pl = []
for α in αs
    Tmax_curve = if α ≤ 0.32; 0.95 elseif α ≤ 0.45; 0.70 elseif α ≤ 0.52; 0.40 else 0.35 end
    p = plot(xlabel="T", ylabel="⟨φ⟩", title=@sprintf("α=%.2f", α),
             xlims=(0, Tmax_curve), ylims=(-0.1, 1.0),
             legend=:bottomleft, titlefontsize=8)
    for (j, N) in enumerate(Ns)
        cold = get(g, (α, N, :cold), nothing)
        col_cold = palette(:viridis, max(length(Ns), 2))[j]
        if cold !== nothing
            T = [t for (t,_) in cold]; φ = [p for (_,p) in cold]
            plot!(p, T, φ, color=col_cold, lw=2, ls=:solid,
                  marker=:circle, ms=3, label="cold N=$N")
        end
    end
    # phi_eq reference
    Ts = 0.0:0.005:Tmax_curve
    plot!(p, Ts, phi_eq.(Ts), color=:gray, lw=1, ls=:dot, label="φ_eq(T)")
    # leading-order spinodal vertical line
    Tsp = T_sp_LO(α)
    if 0 < Tsp ≤ Tmax_curve
        vline!(p, [Tsp], color=:purple, ls=:dot, lw=1.5,
               label="T_sp^LO=$(round(Tsp,digits=3))")
    end
    push!(pl, p)
end
fig = plot(pl..., layout=(2,2), size=(900, 700), dpi=200)
savefig(fig, joinpath(out_dir, "spinodal_cold_hot.png"))
savefig(fig, joinpath(out_dir, "spinodal_cold_hot.pdf"))
println("Saved: $(out_dir)/spinodal_cold_hot.{png,pdf}")

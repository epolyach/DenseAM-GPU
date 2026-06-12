#=
log10(τ) vs T at N=100 from N-dim hybrid MC.
Source: escape_time_Ndim_N100.csv
Output: panels_paper/log_tau_vs_T_N100_Ndim.{png,pdf,eps}
=#

using Printf
using Plots
using Statistics

const N = 100
const CSV_PATH = joinpath(@__DIR__, @sprintf("escape_time_Ndim_N%d.csv", N))
const OUT_DIR  = joinpath(@__DIR__, "panels_paper")
isdir(OUT_DIR) || mkpath(OUT_DIR)

const ALPHA_LIST = [0.50, 0.55, 0.58, 0.60, 0.62]
const ALPHA_COLOR = Dict(
    0.50 => RGB(0.10, 0.20, 0.55),
    0.55 => RGB(0.20, 0.45, 0.75),
    0.58 => RGB(0.40, 0.65, 0.85),
    0.60 => RGB(0.85, 0.45, 0.10),
    0.62 => RGB(0.65, 0.10, 0.20),
)

function load_cells(path)
    cells = Dict{Tuple{Float64,Float64}, Tuple{Vector{Int},Vector{Int}}}()
    max_steps = 0
    for line in eachline(path)
        (isempty(line) || startswith(line, "#") || startswith(line, "alpha")) && continue
        f = split(line, ",")
        α   = parse(Float64, f[1])
        T   = parse(Float64, f[2])
        τ   = parse(Int,     f[5])
        esc = parse(Int,     f[6]) == 1
        key = (round(α, digits=4), round(T, digits=5))
        haskey(cells, key) || (cells[key] = (Int[], Int[]))
        esc ? push!(cells[key][1], τ) : push!(cells[key][2], τ)
        max_steps = max(max_steps, τ)
    end
    return cells, max_steps
end

cells, max_steps = load_cells(CSV_PATH)
@info "max_steps observed = $max_steps"

p = plot(size=(820, 540),
         xlabel="temperature  T",
         ylabel="log10  ⟨τ⟩  (MC steps)",
         legend=:topright,
         framestyle=:box,
         grid=:on, gridalpha=0.25,
         guidefontsize=11, tickfontsize=10)

for α in ALPHA_LIST
    Ts_α = sort([k[2] for k in keys(cells) if k[1] ≈ α])
    isempty(Ts_α) && continue
    T_full = Float64[]; τ_full = Float64[]
    T_cap  = Float64[]; τ_cap  = Float64[]
    for T in Ts_α
        esc, cap = cells[(α, T)]
        if !isempty(esc)
            push!(T_full, T); push!(τ_full, mean(esc))
        else
            push!(T_cap, T);  push!(τ_cap, max_steps)
        end
    end
    col = ALPHA_COLOR[α]
    if !isempty(T_full)
        plot!(p, T_full, log10.(τ_full); lw=2.2, color=col, marker=:circle,
              markersize=5, markerstrokewidth=0,
              label=@sprintf("α = %.2f", α))
    end
end

hline!(p, [log10(max_steps)], color=:gray, ls=:dash, lw=1.2,
       label=@sprintf("ceiling 10^%d", round(Int, log10(max_steps))))

xlims!(p, (0.0, 0.20))

stem = "log_tau_vs_T_N100_Ndim"
out_png = joinpath(OUT_DIR, stem*".png")
out_pdf = joinpath(OUT_DIR, stem*".pdf")
out_eps = joinpath(OUT_DIR, stem*".eps")
savefig(p, out_png); savefig(p, out_pdf)
run(`pdftops -eps $out_pdf $out_eps`)
println("Saved: ", out_png, "  ", out_pdf, "  ", out_eps)

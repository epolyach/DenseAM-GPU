using CSV
using DataFrames
using Plots

# ──────────────── Theoretical boundaries ────────────────

function critical_alpha_lse(T::Real)
    phi = 0.5 * (-T + sqrt(T^2 + 4))
    f_ret = (1 - phi) - (T/2) * log(1 - phi^2)
    alpha_c = 0.5 * (1 - f_ret)^2
    return clamp(alpha_c, 0.0, 0.5)
end

function critical_alpha_lsr(T::Real, b::Real)
    A = b*T + 1
    B = -(2 + T + T*b)
    C = T
    disc = B^2 - 4*A*C
    disc < 0 && return NaN
    y = (-B - sqrt(disc)) / (2*A)
    phi = 1 - y
    (phi <= 1 - 1/b || phi > 1 || phi < 0) && return NaN
    u = -(1/b) * log(1 - b*(1-phi))
    s = 0.5 * log(1 - phi^2)
    f_ret = u - T * s
    alpha_c = 0.5 * (1 - f_ret)^2
    return clamp(alpha_c, 0.0, 0.5)
end

function find_T_max_lsr(b::Real, alpha_th::Real)
    for T in range(0.01, 3.0, length=1000)
        ac = critical_alpha_lsr(T, b)
        !isnan(ac) && ac <= alpha_th && return T
    end
    return NaN
end

# ──────────────── Read data (Parallel Tempering, P 200–5000) ────────────────

data_lse = CSV.read("lse_pt.csv", DataFrame)
data_lsr = CSV.read("lsr_pt.csv", DataFrame)

alpha_values = data_lse[:, 1]
T_columns = names(data_lse)[2:end]
T_values = [parse(Float64, replace(col, "T" => "")) for col in T_columns]

lse_matrix = Matrix(data_lse[:, 2:end])
lsr_matrix = Matrix(data_lsr[:, 2:end])

# ──────────────── Theoretical curves ────────────────

b = 2 + sqrt(2)
alpha_th = 0.5 * (1 - 1/b)^2
T_max = find_T_max_lsr(b, alpha_th)

T_theory = range(0.001, maximum(T_values), length=500)
alpha_c_lse = [critical_alpha_lse(T) for T in T_theory]
alpha_c_lsr = [critical_alpha_lsr(T, b) for T in T_theory]

# ──────────────── Shared settings ────────────────

cmap = cgrad([:darkblue, :blue, :white, :red, :darkred], rev=true)
clims = (minimum([minimum(lse_matrix), minimum(lsr_matrix)]),
         maximum([maximum(lse_matrix), maximum(lsr_matrix)]))

# ──────────────── Left panel: LSE ────────────────

p1 = heatmap(alpha_values, T_values, lse_matrix',
             xlabel="α = ln(P)/N",
             ylabel="T",
             title="LSE  P: 200-5k  PT",
             color=cmap, clims=clims,
             colorbar=false)

plot!(p1, alpha_c_lse, collect(T_theory),
      color=:black, linewidth=2.5,
      label="Theoretical boundary",
      legend=:topright)

# ──────────────── Right panel: LSR ────────────────

p2 = heatmap(alpha_values, T_values, lsr_matrix',
             xlabel="α = ln(P)/N",
             ylabel="",
             title="LSR  P: 200-5k  PT",
             color=cmap, clims=clims,
             colorbar=false)

valid_idx = .!isnan.(alpha_c_lsr) .& (alpha_c_lsr .> alpha_th) .& (alpha_c_lsr .<= 0.5)
plot!(p2, alpha_c_lsr[valid_idx], collect(T_theory)[valid_idx],
      color=:black, linewidth=2.5, label=false)

if !isnan(T_max)
    plot!(p2, [alpha_th, alpha_th], [T_max, maximum(T_values)],
          color=:black, linewidth=2.5, label=false)
end

vline!(p2, [0.5],
       color=:black, linewidth=1.5, linestyle=:dash,
       label=false)

# ──────────────── Colorbar panel ────────────────

cb_ticks = range(clims[1], clims[2], length=100)
cb_data = reshape(cb_ticks, 1, :)
p3 = heatmap([0], collect(cb_ticks), cb_data,
             color=cmap, clims=clims, colorbar=false,
             xticks=false, ylabel="φ",
             framestyle=:box,
             left_margin=0Plots.mm, right_margin=0Plots.mm,
             title="")

# ──────────────── Combine ────────────────

lay = grid(1, 3, widths=[0.485, 0.485, 0.03])
fig = plot(p1, p2, p3, layout=lay, size=(1400, 600),
           left_margin=5Plots.mm, bottom_margin=5Plots.mm)

savefig(fig, "maps1_pt.png")
println("Combined PT plot saved: maps1_pt.png")

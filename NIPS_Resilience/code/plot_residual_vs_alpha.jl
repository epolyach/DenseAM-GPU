#=
Residual plots: ln(τK/τ_rel) - c·ΔF/T vs α for different c values
Output: panels_paper/residual_c=0.25_vs_alpha.{png,pdf}
        panels_paper/residual_c=0.50_vs_alpha.{png,pdf}
=#

using Plots, Printf, LaTeXStrings, SpecialFunctions, LinearAlgebra
default(guidefontsize=8, tickfontsize=7, legendfontsize=6)

const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = FIG_W
const M_PAT = 20000
const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr

function φ_eq_LSR(T)
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b_lsr + b_lsr*φ; D ≤ 1e-10 && (φ = φ_c + 0.005; continue)
        f = (1 - φ^2) - T*φ*D; fp = -2φ - T*(D + b_lsr*φ)
        φ = clamp(φ - f/fp, φ_c + 1e-8, 1 - 1e-8)
    end; return φ
end

function φ_min_val(T)
    φeq = φ_eq_LSR(T)
    return φ_c * φeq - sqrt((1 - φ_c^2) * (1 - φeq^2))
end

function compute_K(α, T)
    N = log(M_PAT) / α; φeq = φ_eq_LSR(T)
    pm = max(0.0, φ_min_val(T))
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
    n_pts = 1000; dφ = (1.0 - pm) / n_pts; dφ ≤ 0 && return 0.0
    integral = 0.0
    for i in 0:n_pts
        φ = pm + i * dφ; φ ≥ 1 && break
        s = 1 - φ^2; s ≤ 0 && continue
        integral += exp(logC + (N-3)/2 * log(s)) * dφ
    end
    return (M_PAT - 1) * integral
end

function barrier_at_phi1max(α, T)
    N = log(M_PAT) / α; φeq = φ_eq_LSR(T)
    R2 = 1 - φeq^2; R2 ≤ 0 && return Inf
    φ1max = sqrt(1 - exp(-2α))
    v_entry = (φ_c - φeq * φ1max) / sqrt(1 - φ1max^2)
    v_entry ≤ 0 && return 0.0
    v2R2 = v_entry^2 / R2
    v2R2 ≥ 0.9999 && return Inf
    return (N-3)/2 * (-log(1 - v2R2))
end

v13_lines = readlines(joinpath(@__DIR__, "v13_diffusion.csv"))
v13_α = Float64[]; v13_T = Float64[]; v13_Dv = Float64[]
for line in v13_lines[2:end]
    f = split(line, ","); length(f) < 6 && continue
    push!(v13_α, parse(Float64, f[1])); push!(v13_T, parse(Float64, f[2]))
    push!(v13_Dv, parse(Float64, f[6]))
end
v13_alphas = sort(unique(v13_α))

function interp_Dv(α_q, T_q)
    _, ia = findmin(abs.(v13_alphas .- α_q))
    mask = v13_α .== v13_alphas[ia]
    Ts = v13_T[mask]; Ds = v13_Dv[mask]
    p = sortperm(Ts); Ts = Ts[p]; Ds = Ds[p]
    Dv_peak, i_peak = findmax(Ds)
    T_q > Ts[i_peak] && return Dv_peak
    T_q = clamp(T_q, Ts[1], Ts[i_peak])
    for i in 1:length(Ts)-1
        Ts[i] ≤ T_q ≤ Ts[i+1] || continue
        return Ds[i] + (T_q-Ts[i])/(Ts[i+1]-Ts[i])*(Ds[i+1]-Ds[i])
    end; return Ds[end]
end

function tau_rel(α, T)
    φeq = φ_eq_LSR(T); R2 = 1 - φeq^2
    Dv = interp_Dv(α, T)
    return R2 / Dv
end

function load_tau_data()
    dir = @__DIR__
    file_map = Dict{Tuple{Float64,Float64}, String}()
    priority = Dict("v17" => 3, "v16b" => 2, "v16" => 1)
    for f in readdir(dir)
        for pfx in ["v17_Pesc_a", "v16b_Pesc_a", "v16_Pesc_a"]
            if startswith(f, pfx) && endswith(f, ".csv")
                m_a = match(r"a(\d+\.\d+)", f); m_T = match(r"T(\d+\.\d+)", f)
                m_a === nothing && continue; m_T === nothing && continue
                α = parse(Float64, m_a.captures[1]); T = parse(Float64, m_T.captures[1])
                α < 0.20 && continue
                ver = pfx[1:end-7]
                key = (α, T)
                old_ver = haskey(file_map, key) ? split(file_map[key],"_")[1] : ""
                if !haskey(file_map, key) || priority[ver] > get(priority, old_ver, 0)
                    file_map[key] = f
                end
            end
        end
    end
    results = Tuple{Float64,Float64,Float64}[]
    for ((α, T), f) in file_map
        lines = readlines(joinpath(dir, f))
        steps = Float64[]; pesc = Float64[]
        for l in lines[2:end]
            parts = split(l, ","); length(parts) < 2 && continue
            push!(steps, parse(Float64, parts[1])); push!(pesc, parse(Float64, parts[2]))
        end
        surv = max.(1.0 .- pesc, 1e-10)
        t0 = 0.0
        for p_hi in [0.10, 0.20, 0.35, 0.50]
            mask = (pesc .> 0.003) .& (pesc .< p_hi)
            sum(mask) < 4 && continue
            t = steps[mask]; y = -log.(1.0 .- pesc[mask])
            X = hcat(ones(length(t)), t); β = X \ y
            if β[2] > 0; t0 = max(0.0, -β[1]/β[2]); break; end
        end
        mask_fit = (surv .> 0.005) .& (steps .> t0)
        sum(mask_fit) < 5 && continue
        t_m = steps[mask_fit]; lnS = log.(surv[mask_fit])
        tau_guess = steps[end] / 5
        best_K = NaN; best_tch = NaN; best_err = Inf
        for lr in range(-2, 2, length=400)
            tch = tau_guess * 10^lr
            u = @. 1 - exp(-(t_m - t0) / tch)
            K = -dot(u, lnS) / dot(u, u)
            K ≤ 0 && continue
            err = sum(@. (lnS + K*u)^2)
            if err < best_err; best_err = err; best_K = K; best_tch = tch; end
        end
        isnan(best_K) && continue
        τ = best_tch / best_K
        push!(results, (α, T, τ))
    end
    return results
end

# ── Main ──
out_dir = joinpath(@__DIR__, "..", "panels_paper"); mkpath(out_dir)

println("Loading data...")
data = load_tau_data()
sort!(data, by=x->(x[1], x[2]))
@printf("Loaded %d points\n", length(data))

T_all_sorted = sort(unique([T for (_, T, _) in data]))

dist_colors = [:red, :blue, :green, :orange, :purple, :black,
               :cyan, :magenta, :brown, :olive, :navy, :crimson]
dist_styles = [:solid, :solid, :solid, :solid, :solid, :solid,
               :dash, :dash, :dash, :dash, :dash, :dash]
dist_markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :pentagon,
                :circle, :square, :diamond, :utriangle, :dtriangle, :pentagon]

function plot_residual(c_val, data, T_all_sorted)
    p = plot(xlabel=L"\alpha",
        ylabel=latexstring("\\ln(\\tau K / \\tau_{\\mathrm{rel}}) - $(c_val)\\,\\Delta F/T"),
        xlims=(0.18, 0.36), size=(FIG_W, FIG_H), dpi=FIG_DPI,
        framestyle=:box, legend=:outerright)

    for (i, T_target) in enumerate(T_all_sorted)
        αs = Float64[]; ys = Float64[]
        for (α, T, τ) in data
            abs(T - T_target) < 0.01 || continue
            K = compute_K(α, T); K < 0.01 && continue
            dF = barrier_at_phi1max(α, T); isinf(dF) && continue
            τr = tau_rel(α, T)
            push!(αs, α); push!(ys, log(τ * K / τr) - c_val * dF)
        end
        perm = sortperm(αs)
        length(perm) > 0 && plot!(p, αs[perm], ys[perm],
            lw=1.5, ls=dist_styles[i], marker=dist_markers[i], markersize=3,
            color=dist_colors[i],
            label=@sprintf("%.2f", T_target))
    end
    return p
end

for c_val in [0.25, 0.30, 0.33, 0.50]
    @printf("Plotting residual c=%.2f...\n", c_val)
    p = plot_residual(c_val, data, T_all_sorted)
    fname = @sprintf("residual_c=%.2f_vs_alpha", c_val)
    for ext in ("png", "pdf")
        savefig(p, joinpath(out_dir, "$fname.$ext"))
    end
    @printf("Saved: panels_paper/%s.{png,pdf}\n", fname)
end

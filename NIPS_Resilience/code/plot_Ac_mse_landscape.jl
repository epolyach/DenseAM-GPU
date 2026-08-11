#=
MSE landscape in (log₁₀A, c) space from v17 survival curves
Output: panels_paper/Ac_mse_landscape.{png,pdf}
=#

using Plots, Printf, LaTeXStrings, SpecialFunctions
default(guidefontsize=8, tickfontsize=7)

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

function cp_lnS(t, α, T, A, c)
    N = log(M_PAT)/α; φeq = φ_eq_LSR(T); R2 = 1-φeq^2; R2 ≤ 1e-10 && return 0.0
    Dv = interp_Dv(α, T); τ_rel = R2/Dv; rate_pre = A/τ_rel
    φ_min = max(0.0, φ_c*(φeq-sqrt(R2))) + 1e-8
    φ_zb = φ_c/φeq; φ_hi = min(φ_zb, 0.9999)
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
    integral = 0.0
    if φ_hi > φ_min
        n_φ = 150; dφ = (φ_hi-φ_min)/n_φ
        for i in 0:n_φ
            φ = φ_min + i*dφ; s = 1-φ^2; s ≤ 0 && continue
            v = (φ_c - φeq*φ)/sqrt(s)
            fv = exp(logC + (N-3)/2*log(s))
            if v ≤ 0; λ = rate_pre
            else
                v2R2 = v^2/R2; v2R2 ≥ 0.9999 && continue
                λ = rate_pre * exp(-c*(N-3)/2*(-log(1-v2R2)))
            end
            integral += (1-exp(-λ*t)) * fv * dφ
        end
    end
    if φ_zb < 0.9999
        dφ2 = (0.9999-φ_zb)/20
        for i in 0:20
            φ = φ_zb + i*dφ2; s = 1-φ^2; s ≤ 0 && continue
            integral += (1-exp(-rate_pre*t)) * exp(logC + (N-3)/2*log(s)) * dφ2
        end
    end
    return -(M_PAT-1) * integral
end

struct Panel; α::Float64; T::Float64; t0::Float64; steps::Vector{Float64}; lnS::Vector{Float64}; end

function load_panels()
    dir = @__DIR__; panels = Panel[]
    for f in sort(filter(f -> startswith(f, "v17_Pesc_a") && endswith(f, ".csv"), readdir(dir)))
        m_a = match(r"a(\d+\.\d+)", f); m_T = match(r"T(\d+\.\d+)", f)
        m_a === nothing && continue; m_T === nothing && continue
        α = parse(Float64, m_a.captures[1]); T = parse(Float64, m_T.captures[1])
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
        stride = max(1, length(steps) ÷ 150)
        idx = 1:stride:length(steps)
        mask = (surv[idx] .> 0.01) .& (surv[idx] .< 0.99)
        sum(mask) < 3 && continue
        sel = idx[mask]
        push!(panels, Panel(α, T, t0, steps[sel], log.(surv[sel])))
    end
    return panels
end

function eval_mse(A, c, panels)
    total = 0.0; n = 0
    for pd in panels
        for i in eachindex(pd.steps)
            t_eff = max(0.0, pd.steps[i] - pd.t0)
            lnS_m = cp_lnS(t_eff, pd.α, pd.T, A, c)
            total += (pd.lnS[i] - lnS_m)^2; n += 1
        end
    end
    return n > 0 ? total / n : Inf
end

function main()
    panels = load_panels()
    @printf("Loaded %d panels\n", length(panels))

    log10A_range = range(-3.0, 0.0, length=15)
    c_range = range(0.05, 0.60, length=15)
    mse_grid = fill(NaN, length(c_range), length(log10A_range))

    for (ic, c) in enumerate(c_range)
        for (ia, log10A) in enumerate(log10A_range)
            A = 10.0^log10A
            mse_grid[ic, ia] = log10(max(eval_mse(A, c, panels), 1e-10))
        end
        @printf("  c=%.2f done\n", c)
    end

    # Find min
    min_val = Inf; min_ia = 1; min_ic = 1
    for ic in 1:length(c_range), ia in 1:length(log10A_range)
        if !isnan(mse_grid[ic, ia]) && mse_grid[ic, ia] < min_val
            min_val = mse_grid[ic, ia]; min_ia = ia; min_ic = ic
        end
    end

    out_dir = joinpath(@__DIR__, "..", "panels_paper"); mkpath(out_dir)

    p = heatmap(collect(log10A_range), collect(c_range), mse_grid,
        color=cgrad(:viridis, rev=true),
        clims=(min_val, min_val + 1.5),
        xlabel=L"\log_{10} A", ylabel=L"c",
        size=(FIG_W, FIG_H), dpi=FIG_DPI,
        framestyle=:box,
        colorbar_title=L"\log_{10}\mathrm{MSE}")

    scatter!(p, [log10A_range[min_ia]], [c_range[min_ic]],
        markersize=8, color=:red, markerstrokecolor=:white, markerstrokewidth=2,
        label=@sprintf("A=%.2e c=%.2f", 10^log10A_range[min_ia], c_range[min_ic]))

    for ext in ("png", "pdf")
        savefig(p, joinpath(out_dir, "Ac_mse_landscape.$ext"))
    end
    println("Saved: panels_paper/Ac_mse_landscape.{png,pdf}")
end

main()

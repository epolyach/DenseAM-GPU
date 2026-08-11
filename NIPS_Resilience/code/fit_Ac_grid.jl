#=
Grid search for (A, c) on v17 survival curves
Then compare τ_data vs τ_model for each panel
=#

using Printf, LinearAlgebra, SpecialFunctions

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
    end
    return φ
end

# D_v from v13
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
    end
    return Ds[end]
end

function cp_lnS(t, α, T, A, c)
    N = log(M_PAT)/α; φeq = φ_eq_LSR(T); R2 = 1-φeq^2; R2 ≤ 1e-10 && return 0.0
    Dv = interp_Dv(α, T); τ_rel = R2/Dv; rate_pre = A/τ_rel
    φ_min = max(0.0, φ_c*(φeq-sqrt(R2))) + 1e-8
    φ_zb = φ_c/φeq; φ_hi = min(φ_zb, 0.9999)
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
    integral = 0.0
    if φ_hi > φ_min
        n_φ = 200; dφ = (φ_hi-φ_min)/n_φ
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
        dφ2 = (0.9999-φ_zb)/30
        for i in 0:30
            φ = φ_zb + i*dφ2; s = 1-φ^2; s ≤ 0 && continue
            integral += (1-exp(-rate_pre*t)) * exp(logC + (N-3)/2*log(s)) * dφ2
        end
    end
    return -(M_PAT-1) * integral
end

# Load panels
struct Panel
    α::Float64; T::Float64; t0::Float64
    steps::Vector{Float64}; lnS::Vector{Float64}
end

function load_panels()
    dir = @__DIR__
    panels = Panel[]
    for f in sort(filter(f -> (startswith(f, "v16_Pesc_a") || startswith(f, "v16b_Pesc_a") || startswith(f, "v17_Pesc_a")) && endswith(f, ".csv"), readdir(dir)))
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
        stride = max(1, length(steps) ÷ 200)
        idx = 1:stride:length(steps)
        mask = (surv[idx] .> 0.01) .& (surv[idx] .< 0.99)
        sum(mask) < 3 && continue
        sel = idx[mask]
        push!(panels, Panel(α, T, t0, steps[sel], log.(surv[sel])))
    end
    return panels
end

function chi2(A, c, panels)
    total = 0.0; n = 0
    for pd in panels
        for i in eachindex(pd.steps)
            t_eff = max(0.0, pd.steps[i] - pd.t0)
            lnS_m = cp_lnS(t_eff, pd.α, pd.T, A, c)
            total += (pd.lnS[i] - lnS_m)^2; n += 1
        end
    end
    return total / n
end

function tau_from_model(α, T, A, c)
    dt = 1.0
    lnS0 = cp_lnS(0.0, α, T, A, c)
    lnS1 = cp_lnS(dt, α, T, A, c)
    λ = -(lnS1 - lnS0) / dt
    return λ > 0 ? 1.0/λ : Inf
end

function tau_from_data(pd)
    steps = pd.steps; surv = exp.(pd.lnS)
    pesc = 1.0 .- surv
    t0 = pd.t0
    mask_fit = (surv .> 0.005) .& (steps .> t0)
    sum(mask_fit) < 5 && return NaN
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
    isnan(best_K) ? NaN : best_tch / best_K
end

# ── Main ──
function main()
    panels = load_panels()
    @printf("Loaded %d panels\n", length(panels))

    # Grid search
    @printf("\nGrid search:\n")
    best_A = 0.0; best_c = 0.0; best_mse = Inf
    for log10A in range(-3.0, 0.0, length=20)
        for c in range(0.05, 0.60, length=20)
            A = 10.0^log10A
            mse = chi2(A, c, panels)
            if mse < best_mse
                best_mse = mse; best_A = A; best_c = c
            end
        end
        @printf("  log10A=%.2f done (best so far: A=%.3e c=%.3f MSE=%.6f)\n",
                log10A, best_A, best_c, best_mse)
    end
    @printf("\n═══ BEST: A = %.4e, c = %.4f, MSE = %.6f (RMS lnS = %.4f) ═══\n",
            best_A, best_c, best_mse, sqrt(best_mse))

    # τ comparison
    @printf("\nτ comparison (A=%.4e, c=%.4f):\n", best_A, best_c)
    @printf("  %-5s %-5s %10s %10s %6s\n", "α", "T", "τ_data", "τ_model", "ratio")
    for pd in panels
        τd = tau_from_data(pd)
        τm = tau_from_model(pd.α, pd.T, best_A, best_c)
        ratio = isnan(τd) ? NaN : τd/τm
        @printf("  %.2f  %.2f  %10.0f  %10.0f  %6.2f\n", pd.α, pd.T, τd, τm, ratio)
    end

    # ── Fine grid on metastable panels only ──
    meta_panels = filter(pd -> tau_from_data(pd) > 41000, panels)
    @printf("\n═══ Fine grid on %d metastable panels ═══\n", length(meta_panels))
    best_Am = 0.0; best_cm = 0.0; best_msem = Inf
    for log10A in range(-4.0, 0.0, length=60)
        for c in range(0.10, 0.60, length=60)
            A = 10.0^log10A
            mse = chi2(A, c, meta_panels)
            if mse < best_msem
                best_msem = mse; best_Am = A; best_cm = c
            end
        end
    end
    @printf("BEST metastable: A = %.4e, c = %.4f, MSE = %.6f (RMS lnS = %.4f)\n",
            best_Am, best_cm, best_msem, sqrt(best_msem))

    @printf("\nτ comparison (A=%.4e, c=%.4f) metastable:\n", best_Am, best_cm)
    @printf("  %-5s %-5s %10s %10s %6s\n", "α", "T", "τ_data", "τ_model", "ratio")
    for pd in meta_panels
        τd = tau_from_data(pd)
        τm = tau_from_model(pd.α, pd.T, best_Am, best_cm)
        ratio = isnan(τd) ? NaN : τd/τm
        @printf("  %.2f  %.2f  %10.0f  %10.0f  %6.2f\n", pd.α, pd.T, τd, τm, ratio)
    end

    # ── c=1 on metastable panels only ──
    @printf("\n═══ c=1 on %d metastable panels ═══\n", length(meta_panels))
    best_A1m = 0.0; best_mse1m = Inf
    for log10A in range(-4.0, 0.0, length=200)
        A = 10.0^log10A
        mse = chi2(A, 1.0, meta_panels)
        if mse < best_mse1m; best_mse1m = mse; best_A1m = A; end
    end
    @printf("BEST c=1 metastable: A = %.4e, MSE = %.6f (RMS lnS = %.4f)\n",
            best_A1m, best_mse1m, sqrt(best_mse1m))

    # Also test A = 0.018 with c=1
    mse_018 = chi2(0.018, 1.0, meta_panels)
    @printf("A=0.018, c=1: MSE = %.6f (RMS = %.4f)\n", mse_018, sqrt(mse_018))

    @printf("\nτ comparison (A=0.018, c=1.0) metastable:\n")
    @printf("  %-5s %-5s %10s %10s %6s\n", "α", "T", "τ_data", "τ_model", "ratio")
    for pd in sort(meta_panels, by=p->(p.α, p.T))
        τd = tau_from_data(pd)
        τm = tau_from_model(pd.α, pd.T, 0.018, 1.0)
        ratio = isnan(τd) ? NaN : τd/τm
        @printf("  %.2f  %.2f  %10.0f  %10.0f  %6.2f\n", pd.α, pd.T, τd, τm, ratio)
    end
end

main()

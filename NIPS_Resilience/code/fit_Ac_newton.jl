#=
Newton-Raphson fit of (A, c) from v17 S(t) survival curves
────────────────────────────────────────────────────────────────────────
Fits compound Poisson S(t-t₀) directly to survival data.
t₀ determined individually per panel from the data.
Objective: Σ_panels Σ_t (ln S_data - ln S_model)²
────────────────────────────────────────────────────────────────────────
=#

using Printf, LinearAlgebra, SpecialFunctions

const M_PAT = 20000
const b_lsr_val = 2 + sqrt(2)
const φ_c_val = (b_lsr_val - 1) / b_lsr_val

function φ_eq_LSR(T)
    b = b_lsr_val; pc = φ_c_val
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b + b*φ; D ≤ 1e-10 && (φ = pc + 0.005; continue)
        f = (1 - φ^2) - T*φ*D; fp = -2φ - T*(D + b*φ)
        φ = clamp(φ - f/fp, pc + 1e-8, 1 - 1e-8)
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

# Compound Poisson ln S(t) for a single time
function cp_lnS(t, α, T, A, c)
    N = log(M_PAT) / α
    φeq = φ_eq_LSR(T); R2 = 1-φeq^2; R2 ≤ 1e-10 && return 0.0
    Dv = interp_Dv(α, T); τ_rel = R2/Dv; rate_pre = A/τ_rel
    φ_min = max(0.0, φ_c_val*(φeq-sqrt(R2))) + 1e-8
    φ_zb = φ_c_val/φeq; φ_hi = min(φ_zb, 0.9999)
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
    integral = 0.0
    if φ_hi > φ_min
        n_φ = 300; dφ = (φ_hi-φ_min)/n_φ
        for i in 0:n_φ
            φ = φ_min + i*dφ; s = 1-φ^2; s ≤ 0 && continue
            v = (φ_c_val - φeq*φ)/sqrt(s)
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
        dφ2 = (0.9999-φ_zb)/50
        for i in 0:50
            φ = φ_zb + i*dφ2; s = 1-φ^2; s ≤ 0 && continue
            integral += (1-exp(-rate_pre*t)) * exp(logC + (N-3)/2*log(s)) * dφ2
        end
    end
    return -(M_PAT-1) * integral
end

# Panel data
struct Panel
    α::Float64; T::Float64
    t0::Float64           # individual lag
    steps::Vector{Float64}  # time points (after filtering)
    lnS::Vector{Float64}    # ln S(t) data
end

function load_panels()
    dir = @__DIR__
    files = sort(filter(f -> startswith(f, "v17_Pesc_a") && endswith(f, ".csv"), readdir(dir)))
    panels = Panel[]
    for f in files
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

        # Determine t₀: fit -ln(1-P) = λ(t-t₀) on early data
        t0 = 0.0
        for p_hi in [0.10, 0.20, 0.35, 0.50]
            mask = (pesc .> 0.003) .& (pesc .< p_hi)
            sum(mask) < 4 && continue
            t = steps[mask]; y = -log.(1.0 .- pesc[mask])
            X = hcat(ones(length(t)), t)
            β = X \ y
            if β[2] > 0
                t0 = max(0.0, -β[1]/β[2])
                break
            end
        end

        # Filter: use S ∈ [0.01, 0.99], stride for speed
        stride = max(1, length(steps) ÷ 300)
        idx = 1:stride:length(steps)
        mask = (surv[idx] .> 0.01) .& (surv[idx] .< 0.99)
        sum(mask) < 3 && continue
        sel = idx[mask]

        push!(panels, Panel(α, T, t0, steps[sel], log.(surv[sel])))
        @printf("  α=%.2f T=%.2f  t₀=%6.0f  n_pts=%d\n", α, T, t0, length(sel))
    end
    return panels
end

# χ² = mean over all panels and points of (ln S_data - ln S_model)²
function chi2(logA, c, panels)
    A = exp(logA)
    total = 0.0; n_pts = 0
    for pd in panels
        for i in eachindex(pd.steps)
            t_eff = max(0.0, pd.steps[i] - pd.t0)
            lnS_m = cp_lnS(t_eff, pd.α, pd.T, A, c)
            total += (pd.lnS[i] - lnS_m)^2
            n_pts += 1
        end
    end
    return total / n_pts
end

# ── Load data ──
println("Loading v17 panels...")
panels = load_panels()
@printf("Total: %d panels\n", length(panels))

# ── Newton-Raphson with LM damping ──
function run_newton(panels)
    logA = log(0.031); c = 0.25
    ε = 5e-3  # finite difference step (larger for S(t) objective)
    μ = 0.01

    @printf("\nStart: A=%.4e, c=%.4f, χ²=%.6f\n", exp(logA), c, chi2(logA, c, panels))

    for iter in 1:50
        f0 = chi2(logA, c, panels)
        g1 = (chi2(logA+ε, c, panels) - chi2(logA-ε, c, panels)) / (2ε)
        g2 = (chi2(logA, c+ε, panels) - chi2(logA, c-ε, panels)) / (2ε)
        H11 = (chi2(logA+ε, c, panels) - 2f0 + chi2(logA-ε, c, panels)) / ε^2
        H22 = (chi2(logA, c+ε, panels) - 2f0 + chi2(logA, c-ε, panels)) / ε^2
        H12 = (chi2(logA+ε, c+ε, panels) - chi2(logA+ε, c-ε, panels) -
               chi2(logA-ε, c+ε, panels) + chi2(logA-ε, c-ε, panels)) / (4ε^2)
        H = [H11+μ H12; H12 H22+μ]
        g = [g1; g2]
        det_H = H[1,1]*H[2,2] - H[1,2]^2
        if abs(det_H) < 1e-30; μ *= 10; continue; end
        δ = H \ g

        # Line search
        best_s = 0.0; best_f = f0
        for s in [1.0, 0.5, 0.25, 0.1, 0.05]
            la = logA - s*δ[1]; cc = max(0.01, c - s*δ[2])
            ft = chi2(la, cc, panels)
            if ft < best_f; best_f = ft; best_s = s; end
        end
        if best_s == 0
            μ *= 5
            @printf("  iter %2d: no improvement, μ=%.1e\n", iter, μ)
            μ > 1e8 && break
            continue
        end
        logA -= best_s * δ[1]; c = max(0.01, c - best_s * δ[2])
        μ = max(μ * 0.5, 1e-6)
        f_new = chi2(logA, c, panels)
        @printf("  iter %2d: A=%.4e c=%.4f χ²=%.6f (s=%.2f μ=%.1e Δ=%.2e)\n",
                iter, exp(logA), c, f_new, best_s, μ, f0-f_new)
        abs(f_new - f0) < 1e-8 * abs(f0) && (@printf("  Converged!\n"); break)
    end
    return logA, c
end

logA, c = run_newton(panels)

# Errors
ε = 5e-3
f0 = chi2(logA, c, panels)
H11 = (chi2(logA+ε, c, panels) - 2f0 + chi2(logA-ε, c, panels)) / ε^2
H22 = (chi2(logA, c+ε, panels) - 2f0 + chi2(logA, c-ε, panels)) / ε^2
H12 = (chi2(logA+ε, c+ε, panels) - chi2(logA+ε, c-ε, panels) -
       chi2(logA-ε, c+ε, panels) + chi2(logA-ε, c-ε, panels)) / (4ε^2)
H = [H11 H12; H12 H22]
det_H = H[1,1]*H[2,2] - H[1,2]^2
n_pts = sum(length(pd.steps) for pd in panels)

@printf("\n═══ RESULT ═══\n")
@printf("  A = %.4e\n", exp(logA))
@printf("  c = %.4f\n", c)
@printf("  χ²/n = %.6f (RMS ln S = %.4f)\n", f0, sqrt(f0))
@printf("  n_panels = %d, n_pts = %d\n", length(panels), n_pts)
if det_H > 0
    Hinv = [H[2,2] -H[1,2]; -H[1,2] H[1,1]] / det_H
    cov = (2f0/n_pts) * Hinv
    σ_logA = sqrt(max(0, cov[1,1])); σ_c = sqrt(max(0, cov[2,2]))
    ρ = cov[1,2] / (σ_logA*σ_c + 1e-30)
    @printf("  σ(A)/A = %.2f%%\n", 100*σ_logA)
    @printf("  σ(c) = %.4f\n", σ_c)
    @printf("  ρ(logA, c) = %.3f\n", ρ)
else
    @printf("  Hessian not positive definite\n")
end

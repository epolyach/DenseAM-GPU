#=
Smart-MC validity window vs the LSE retrieval boundary (AAAI paper)
────────────────────────────────────────────────────────────────────────
Theoretical retrieval boundary (exact LSE):
    α_c^E(T) = -(1/2) log(1 - (1 - f_ret(T))^2)
    φ_eq(T)  = (1/2)(-T + √(T²+4))
    f_ret(T) = 1 - φ_eq - (T/2) log(1 - φ_eq²)

Invert to get T_crit(α) — the physical retrieval temperature at load α.

Smart-MC validity at (α, N, K):
    K patterns drawn from the upper tail of the spherical density
       (1+φ)/2 ~ Beta((N-1)/2, (N-1)/2)
    so φ_cut(α,N,K) solves P(φ > φ_cut) = K / exp(αN).
    The MH equilibrium overlap with the champion fluctuates around
       φ_eq(T) ± σ,  σ ≈ √(T/N)
    so the approximation holds while
       φ_eq(T) - 3σ(T,N) ≥ φ_cut(α, N, K).
    Invert this for T → T_safe(α; N, K).

Goal: pick (N, K) curves whose T_safe(α) stays ABOVE T_crit(α) over the
interesting α range — those configurations can faithfully probe the
retrieval boundary with smart MC.

Output: ../panels_paper/smart_MC_validity.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

ENV["GKSwstype"] = "100"

using Plots
using SpecialFunctions
using QuadGK
using Printf

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

const FIG_DPI    = 300
const FIG_W      = round(Int, 86/25.4 * 100) * 2     # double-wide
const FIG_H      = round(Int, 86/25.4 * 100)
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_LEG   = 6

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK, legendfontsize=FONT_LEG)

# ─────────────── Theory: LSE exact boundary ───────────────
φ_eq(T)   = 0.5 * (-T + sqrt(T^2 + 4))
f_ret(T)  = let φ = φ_eq(T); 1 - φ - (T/2)*log(1 - φ^2); end
α_c_exact(T) = let fr = f_ret(T)
    fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5*log(arg)
end

# Invert α_c^E(T) = α  →  T_crit(α)  by bisection
function T_crit(α; T_lo=1e-5, T_hi=2.0)
    α <= 0 && return Inf
    α_c_exact(T_lo) < α && return T_lo
    α_c_exact(T_hi) > α && return T_hi
    for _ in 1:80
        m = 0.5*(T_lo + T_hi)
        α_c_exact(m) > α ? (T_lo = m) : (T_hi = m)
    end
    return 0.5*(T_lo + T_hi)
end

# ─────────────── Smart-MC: φ_cut via Beta((N-1)/2,(N-1)/2) ───────────────
beta_a(N) = (N - 1)/2

function log_beta_sf(x, a)
    x <= 0 && return 0.0
    x >= 1 && return -Inf
    if x < 0.95
        _, q = beta_inc(a, a, x); q > 0 && return log(q)
    end
    u = 1 - x
    integrand(t) = t > 0 ? exp((a-1)*(log(t) + log1p(-u*t))) : 0.0
    val, _ = quadgk(integrand, 0.0, 1.0; rtol=1e-10)
    return a*log(u) + log(val) - logbeta(a, a)
end

function beta_isf_log(log_p, a)
    log_p >= 0 && return 0.0
    log_u = (log_p + log(a) + logbeta(a, a)) / a
    log_u = min(log_u, log(0.5))
    for _ in 1:60
        u = clamp(exp(log_u), 1e-300, 0.5); x = 1 - u
        lp = log_beta_sf(x, a); err = lp - log_p
        abs(err) < 1e-10 && break
        log_pdf = (a-1)*(log(x) + log(u)) - logbeta(a, a)
        dlog = exp(log_pdf + log(u) - lp); dlog < 1e-12 && (dlog = a)
        log_u -= err / dlog
    end
    return 1 - exp(log_u)
end

# φ_cut(α, N, K).  Returns -1 if K ≥ M (no truncation).
function phi_cut(α, N, K)
    log_p = log(K) - α*N
    log_p >= 0 && return -1.0
    return 2*beta_isf_log(log_p, beta_a(N)) - 1
end

# Bare bound:  φ_eq(T_max) = φ_cut  ⇒  T_max = (1-φ_cut^2)/φ_cut
function T_max_bare(α, N, K; T_hi=2.0)
    φc = phi_cut(α, N, K)
    φc <= -1 && return T_hi
    φc <= 0  && return T_hi
    return (1 - φc^2)/φc
end

# Proper Gaussian σ in φ around the saddle:
#   F(φ) = N f(φ),  f(φ) = -φ - (T/2) log(1-φ^2)
#   f''(φ_eq) = T (1+φ_eq^2) / (1-φ_eq^2)^2
#   σ_φ = sqrt( T / (N f''(φ_eq)) ) = (1-φ_eq^2) / sqrt( N (1+φ_eq^2) )
σ_phi(T, N) = let φ = φ_eq(T); (1 - φ^2) / sqrt(N * (1 + φ^2)); end

# Safety-margin bound:  φ_eq(T) - k·σ_φ(T,N) = φ_cut
function T_safe(α, N, K; sigma_k=3.0, T_hi=2.0)
    φc = phi_cut(α, N, K)
    φc <= -1 && return T_hi
    g(T) = φ_eq(T) - sigma_k*σ_phi(T, N) - φc
    g(T_hi) >= 0 && return T_hi
    T_lo = 0.0
    for _ in 1:80
        m = 0.5*(T_lo + T_hi); g(m) >= 0 ? (T_lo = m) : (T_hi = m)
    end
    return 0.5*(T_lo + T_hi)
end

# ─────────────── α grid and curves ───────────────
αs = collect(range(0.30, 1.60, length=400))

# (label, N, K, color, ls)
configs = [
    ("N=15,  K=full",          15,   typemax(Int),       :black,      :solid),
    ("N=25,  K=full",          25,   typemax(Int),       :gray40,     :solid),
    ("N=50,  K=10⁴",           50,   10_000,             :royalblue,  :solid),
    ("N=100, K=10⁴",           100,  10_000,             :seagreen,   :solid),
    ("N=100, K=e¹²",           100,  round(Int, exp(12)),:seagreen,   :dash),
    ("N=200, K=10⁴",           200,  10_000,             :darkorange, :solid),
    ("N=200, K=e¹²",           200,  round(Int, exp(12)),:darkorange, :dash),
    ("N=500, K=10⁴",           500,  10_000,             :firebrick,  :solid),
]

# Theoretical T_crit(α)
Tcrit = [T_crit(α) for α in αs]

println("α-range: $(first(αs)) … $(last(αs))")
@printf("T_crit at α=0.5: %.4f    α=1.0: %.4f    α=1.5: %.4f\n",
        T_crit(0.5), T_crit(1.0), T_crit(1.5))

# ─────────────── PLOT ───────────────
println("Plotting validity diagram...")

# y-limits: low T magnified via log axis
ymin, ymax = 1e-3, 2.0

p = plot(xlabel="α  =  log M / N",
         ylabel="T",
         title="Smart-MC validity window vs. exact LSE retrieval boundary",
         titlefontsize=FONT_GUIDE,
         xlims=(first(αs), last(αs)),
         ylims=(ymin, ymax),
         yscale=:log10,
         legend=:bottomleft,
         size=(FIG_W, FIG_H), dpi=FIG_DPI,
         left_margin=4Plots.mm, bottom_margin=4Plots.mm, top_margin=2Plots.mm,
         right_margin=4Plots.mm)

# Shade the forbidden (no-retrieval) region above T_crit
plot!(p, αs, Tcrit,
      fillrange=fill(ymax, length(αs)),
      fillalpha=0.10, fillcolor=:gray, linealpha=0,
      label=false)

# Theory line
plot!(p, αs, Tcrit, lw=2.5, color=:black,
      label="T_crit(α)  exact LSE retrieval boundary")

# Smart-MC validity curves:
#   solid = bare bound  φ_eq(T) = φ_cut         (the *physical* limit)
#   dotted = 3σ safety  φ_eq(T) - 3σ_φ = φ_cut  (negligibly below)
for (lbl, N, K, col, ls) in configs
    Ts_bare = [T_max_bare(α, N, K) for α in αs]
    Ts_safe = [T_safe(α, N, K)     for α in αs]
    plot!(p, αs, Ts_bare, lw=1.6, color=col, ls=ls, label=lbl)
    plot!(p, αs, Ts_safe, lw=0.8, color=col, ls=:dot, label=false)
end

# Reference verticals
vline!(p, [0.5, 1.0], color=:gray, ls=:dot, lw=0.7, label=false)
annotate!(p, 0.50, ymax*0.7, text("α_c^G(0)=0.5", :left, 6, :gray30))
annotate!(p, 1.00, ymax*0.5, text("α=1",          :left, 6, :gray30))

# Annotate the "good" region
annotate!(p, last(αs)-0.05, ymin*2.5,
          text("smart-MC valid ⇔ curve ABOVE T_crit",
               :right, 7, :gray20))

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "smart_MC_validity.$ext"))
end
println("Saved: panels_paper/smart_MC_validity.{png,pdf}")

# ─────────────── Numerical summary ───────────────
println()
@printf("%-22s  %-9s %-9s %-9s %-9s %-9s\n",
        "config", "α=0.5", "α=0.7", "α=1.0", "α=1.2", "α=1.5")
println("─"^75)
for α in (0.5, 0.7, 1.0, 1.2, 1.5)
    Tc = T_crit(α)
end
for (lbl, N, K, _, _) in configs
    line = @sprintf("%-22s ", lbl)
    for α in (0.5, 0.7, 1.0, 1.2, 1.5)
        Ts = T_max_bare(α, N, K); Tc = T_crit(α)
        flag = Ts > Tc ? "✓" : "✗"
        line *= @sprintf(" %.4f%s", Ts, flag)
    end
    println(line)
end
@printf("\n%-22s  ", "T_crit(α)")
for α in (0.5, 0.7, 1.0, 1.2, 1.5); @printf(" %.4f ", T_crit(α)); end
println("\n\nDone.")

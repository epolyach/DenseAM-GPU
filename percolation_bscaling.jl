#=
Percolation experiments for b = βN scaling
═══════════════════════════════════════════════════════════════
Tests the predictions of Section 3.5: as b grows from O(1) to O(N),
  - φ_c = (b-1)/b → 1
  - α_c = φ_c²/2 shifts from 1/4 toward 1/2
  - barriers become O(1) instead of O(N)
  - exact ⟨K⟩ matches the analytical λ for each b

Panel 1: α_c(b) analytical curve + tested values
Panel 2: λ(α) for several b at N=50
Panel 3: Exact ⟨K⟩ vs analytical λ — validation
Panel 4: Barrier height N/(2b)·ln(1/q) vs b

Usage:
  julia percolation_bscaling.jl                # plot from cached CSV
  julia percolation_bscaling.jl --refresh_csv  # recompute + save CSV

Output: percolation_bscaling.{png,pdf}
═══════════════════════════════════════════════════════════════
=#

using Random, LinearAlgebra, Statistics, Printf, Plots, DelimitedFiles

const REFRESH_CSV = "--refresh_csv" in ARGS
const CSV_FILE = "percolation_bscaling.csv"

# ── Analytical functions (generalized to arbitrary b) ──

φ_c(b) = (b - 1) / b
α_c_anal(b) = φ_c(b)^2 / 2
α_c_perc(b, N) = φ_c(b)^2/2 + log(φ_c(b) * sqrt(2π * N)) / N

function λ_anal(α, N, b)
    qth = φ_c(b)
    log_λ = N * (α - qth^2/2) - log(qth * sqrt(N)) - 0.5*log(2π)
    return exp(log_λ)
end

# ── Spherical vs Gaussian cap probability (numerical) ──

function exact_cap_prob(N, qth; n_quad=50_000)
    # ∫_{qth}^1 (1-q²)^{(N-3)/2} dq  /  ∫_{-1}^1 (1-q²)^{(N-3)/2} dq
    q_hi = range(qth, 1.0, length=n_quad)
    dq = (1.0 - qth) / (n_quad - 1)
    f_hi = [(1 - q^2)^((N-3)/2) for q in q_hi]
    num = dq * (sum(f_hi) - 0.5*(f_hi[1] + f_hi[end]))

    q_all = range(-1.0, 1.0, length=2*n_quad)
    dq2 = 2.0 / (2*n_quad - 1)
    f_all = [(1 - q^2)^((N-3)/2) for q in q_all]
    den = dq2 * (sum(f_all) - 0.5*(f_all[1] + f_all[end]))
    return num / den
end

function gauss_cap_prob(N, qth)
    x = qth * sqrt(N)
    return exp(-x^2/2) / (x * sqrt(2π))
end

correction_factor(N, qth) = exact_cap_prob(N, qth) / gauss_cap_prob(N, qth)

# ── Exact neighbor counting on S^{N-1}(√N) ──

function count_neighbors(N, M, n_real, qth; chunk_size=100_000)
    Ks = zeros(Int, n_real)
    for r in 1:n_real
        K = 0
        remaining = M - 1
        while remaining > 0
            batch = min(remaining, chunk_size)
            z = randn(N, batch)
            norms = vec(sqrt.(sum(z .^ 2, dims=1)))
            qs = vec(z[1, :]) ./ norms
            K += count(>(qth), qs)
            remaining -= batch
        end
        Ks[r] = K
    end
    return Ks
end

# ── Parameters ──

const N_anal = 50   # for analytical panels
const N_exact = 25  # for exact counting (smaller M)
# b values: from LSR-optimal to b = βN regime
# β < 1 and b > 2+√2 ≈ 3.414
const b_values = [2 + sqrt(2), 4.0, 5.0, 10.0, 25.0]
const b_labels = ["3.41 (LSR)", "4", "5", "10", "25 (=N)"]

# ── Compute exact ⟨K⟩ for each b ──

function compute_bscaling(; N=N_exact, max_M=2_000_000, n_real_base=2000)
    println("\n══ Computing b-scaling experiments (N=$N) ══")
    rows = Vector{Vector{Float64}}()

    for (bi, b) in enumerate(b_values)
        qth = φ_c(b)
        ac = α_c_perc(b, N)
        # α range: center around α_c, extend a bit on both sides
        α_lo = max(round(ac - 0.06, digits=2), 0.10)
        α_hi = min(round(ac + 0.12, digits=2), 0.60)
        α_range = α_lo:0.02:α_hi

        @printf("  b=%.2f, φ_c=%.4f, α_c=%.3f, scanning α=[%.2f, %.2f]\n",
                b, qth, ac, α_lo, α_hi)

        for α in α_range
            M = round(Int, exp(N * α))
            M > max_M && continue
            # Fewer realizations for large M
            n_real = M > 500_000 ? 200 : M > 100_000 ? 500 : n_real_base
            Ks = count_neighbors(N, M, n_real, qth)
            Km = mean(Ks)
            λm = λ_anal(α, N, b)
            push!(rows, [b, α, M, qth, λm, Km, Float64(n_real)])
            @printf("    α=%.2f: M=%d, λ=%.4f, ⟨K⟩=%.4f (n=%d)\n",
                    α, M, λm, Km, n_real)
        end
    end

    mat = reduce(hcat, rows)'
    open(CSV_FILE, "w") do io
        println(io, "b,alpha,M,phi_c,lambda_anal,K_mean,n_real")
        writedlm(io, mat, ',')
    end
    println("  → Saved $CSV_FILE")
end

# ── Run if needed ──
if REFRESH_CSV || !isfile(CSV_FILE)
    compute_bscaling()
else
    println("\n  CSV found — plotting from cache. Use --refresh_csv to recompute.")
end

# ── Read data ──
data = readdlm(CSV_FILE, ','; header=true)[1]

# ══════════════════════════════════════════════════════════
# Panel 1: α_c(b) — analytical curve for several N
# ══════════════════════════════════════════════════════════
b_range = range(2 + sqrt(2), 80, length=200)
αc_inf = [α_c_anal(b) for b in b_range]

p1 = plot(xlabel="b", ylabel="α_c",
    title="Percolation threshold α_c(b)",
    legend=:bottomright, legendfontsize=7)
plot!(p1, collect(b_range), αc_inf, lw=2.5, color=:black, label="N→∞")
for (N_plot, ls) in [(25, :dash), (50, :dashdot), (150, :dot)]
    αc_N = [α_c_perc(b, N_plot) for b in b_range]
    plot!(p1, collect(b_range), αc_N, lw=2, ls=ls,
        label=@sprintf("N=%d", N_plot))
end
hline!(p1, [0.25], color=:gray60, ls=:dot, lw=1, label="1/4 (LSR)")
hline!(p1, [0.50], color=:gray60, ls=:dash, lw=1, label="1/2 (exp. AM)")
for b in b_values
    scatter!(p1, [b], [α_c_anal(b)], ms=7, color=:red, label=false,
        markerstrokewidth=1.5, markerstrokecolor=:black)
end

# ══════════════════════════════════════════════════════════
# Panel 2: λ(α) for several b values at N=N_exact
# ══════════════════════════════════════════════════════════
α_full = collect(0.10:0.005:0.60)

p2 = plot(xlabel="α", ylabel="λ  and  ⟨K⟩",
    title=@sprintf("λ(α) and exact ⟨K⟩ (N=%d)", N_exact),
    yscale=:log10, ylims=(1e-3, 1e3),
    legend=:topleft, legendfontsize=6)

for (bi, b) in enumerate(b_values)
    λs = [λ_anal(α, N_exact, b) for α in α_full]
    ac = α_c_perc(b, N_exact)
    plot!(p2, α_full, λs, lw=2, ls=:dash,
        label=@sprintf("λ, b=%s", b_labels[bi]))

    # Overlay exact ⟨K⟩ data
    mask = data[:, 1] .== b
    if any(mask)
        α_emp = data[mask, 2]
        K_emp = data[mask, 6]
        keep = K_emp .> 0
        if any(keep)
            scatter!(p2, α_emp[keep], max.(K_emp[keep], 5e-4),
                ms=5, markershape=:circle, label=@sprintf("⟨K⟩, b=%s", b_labels[bi]))
        end
    end
end
hline!(p2, [1.0], color=:black, ls=:dash, lw=2, label="λ=1")

# ══════════════════════════════════════════════════════════
# Panel 3: Correction factor P_sphere/P_Gauss vs b
# ══════════════════════════════════════════════════════════
b_theory = range(2 + sqrt(2), 30, length=100)
cf_theory = [correction_factor(N_exact, φ_c(b)) for b in b_theory]

p3 = plot(xlabel="b", ylabel="P_sphere / P_Gauss",
    title=@sprintf("Spherical correction (N=%d)", N_exact),
    legend=:topright, legendfontsize=7,
    yscale=:log10, ylims=(1e-4, 2))
plot!(p3, collect(b_theory), cf_theory, lw=2.5, color=:black,
    label="Numerical ratio")
hline!(p3, [1.0], color=:gray60, ls=:dash, lw=1, label="Gaussian = sphere")

# Overlay empirical ⟨K⟩/λ averaged over α for each b
for (bi, b) in enumerate(b_values)
    mask = data[:, 1] .== b
    if !any(mask); continue; end
    λ_emp = data[mask, 5]
    K_emp = data[mask, 6]
    valid = (λ_emp .> 0.1) .& (K_emp .> 0)
    if any(valid)
        ratio = mean(K_emp[valid] ./ λ_emp[valid])
        scatter!(p3, [b], [ratio], ms=8, marker=:diamond,
            markerstrokewidth=1.5, markerstrokecolor=:black,
            label=@sprintf("⟨K⟩/λ, b=%s", b_labels[bi]))
    end
end
annotate!(p3, 15.0, 2e-3, text("sphere tail\nlighter than\nGaussian", 8, :gray40))

# ══════════════════════════════════════════════════════════
# Panel 4: Barrier height scaling
# ══════════════════════════════════════════════════════════
N_barrier = 50

# Correct barrier: ΔE = (N/b) ln(1/(b(φ*_μ - φ_c)))
# where φ*_μ = q·φ_c + √((1-q²)(1-φ_c²))
function barrier_exact(q, N, b)
    qc = φ_c(b)
    φ_star = q * qc + sqrt((1 - q^2) * (1 - qc^2))
    Δφ = φ_star - qc
    Δφ ≤ 0 && return Inf
    return (N / b) * log(1 / (b * Δφ))
end

p4 = plot(xlabel="q (mutual overlap)",
    ylabel="ΔE = (N/b)·ln(1/[b(φ*−φ_c)])",
    title=@sprintf("Energy barrier (N=%d)", N_barrier),
    legend=:topright, legendfontsize=7,
    ylims=(0, 25))

for (bi, b) in enumerate(b_values)
    qgeom = max(2φ_c(b)^2 - 1, 0.0)
    q_barrier = collect(range(max(qgeom + 0.005, 0.01), 0.99, length=200))
    ΔE = [barrier_exact(q, N_barrier, b) for q in q_barrier]
    plot!(p4, q_barrier, ΔE, lw=2,
        label=@sprintf("b=%s (N/b=%.1f)", b_labels[bi], N_barrier/b))
end
hline!(p4, [1.0], color=:red, ls=:dot, lw=1.5, label="T ~ 1")
annotate!(p4, 0.55, 22.0, text("O(N) barriers\n(b = O(1))", 8, :gray40))
annotate!(p4, 0.55, 1.5, text("O(1) barriers\n(b = O(N))", 8, :gray40))

# ══════════════════════════════════════════════════════════
# Combine
# ══════════════════════════════════════════════════════════
p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1000), dpi=150,
    plot_title="Percolation with b-scaling: from LSR (b=3.41) to exponential AM (b=N)",
    margin=5Plots.mm)

savefig(p, "percolation_bscaling.png")
savefig(p, "percolation_bscaling.pdf")
println("\n✓ Saved: percolation_bscaling.{png,pdf}")

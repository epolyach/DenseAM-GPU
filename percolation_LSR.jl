#=
Basin Percolation Test for LSR Model
═════════════════════════════════════════════════════════════════
Tests the hypothesis that the retrieval → paramagnetic transition
corresponds to percolation of pattern basins on S^{N-1}(√N).

Panel 1: Analytical λ(α) for several N — shows percolation threshold
Panel 2: Direct neighbor count K vs Poisson(λ) — validates model
Panel 3: Branching process survival — shows sharp transition
Panel 4: BFS depth-1 vs depth-2 — tests chain termination at depth 1

Usage:
  julia percolation_LSR.jl                  # plot from cached CSV
  julia percolation_LSR.jl --refresh_csv    # recompute + save CSV, then plot

Output: percolation_LSR.{png,pdf}
═════════════════════════════════════════════════════════════════
=#

using Random, LinearAlgebra, Statistics, Printf, Plots, DelimitedFiles

# ──────────────── CLI ────────────────
const REFRESH_CSV = "--refresh_csv" in ARGS

# ──────────────── Constants ────────────────
const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr       # = 1/√2 ≈ 0.7071
const α_∞ = φ_c^2 / 2                  # = 0.25  (N → ∞ limit)
const CORR_COEFF = 0.083               # higher-order correction coefficient

# CSV file paths
const CSV_PANEL2 = "percolation_LSR_panel2.csv"
const CSV_PANEL3 = "percolation_LSR_panel3.csv"
const CSV_PANEL4 = "percolation_LSR_panel4.csv"

@printf("LSR: b = %.4f,  φ_c = %.6f,  α_∞ = φ_c²/2 = %.4f\n", b_lsr, φ_c, α_∞)

# ──────────────── Analytical functions ────────────────

function λ_anal(α, N; q_th=φ_c)
    log_λ = N * (α - q_th^2/2) - log(q_th * sqrt(N)) - 0.5*log(2π)
    return exp(log_λ)
end

α_c_perc(N; q_th=φ_c) = q_th^2/2 + log(q_th * sqrt(2π * N)) / N

function rand_poisson(λ)
    λ ≤ 0 && return 0
    if λ < 30
        L = exp(-λ); k = 0; p = 1.0
        while p > L; k += 1; p *= rand(); end
        return k - 1
    else
        return max(0, round(Int, λ + sqrt(λ) * randn()))
    end
end

# ──────────────── Computation functions ────────────────

function count_neighbors_chunked(N, M, n_real; chunk_size=100_000)
    Ks = zeros(Int, n_real)
    for r in 1:n_real
        K = 0
        remaining = M - 1
        while remaining > 0
            batch = min(remaining, chunk_size)
            z = randn(N, batch)
            norms = vec(sqrt.(sum(z .^ 2, dims=1)))
            qs = vec(z[1, :]) ./ norms
            K += count(>(φ_c), qs)
            remaining -= batch
        end
        Ks[r] = K
    end
    return Ks
end

function compute_panel2(; max_M=2_000_000)
    println("\n══ Computing Panel 2: ⟨K⟩ vs α ══")
    rows = Vector{Vector{Float64}}()
    for N2 in [25, 50]
        n_real2 = N2 == 25 ? 2000 : 500
        for α in 0.24:0.02:0.50
            M = round(Int, exp(N2 * α))
            M > max_M && break
            Ks = count_neighbors_chunked(N2, M, n_real2)
            Km = mean(Ks)
            λm = λ_anal(α, N2)
            push!(rows, [N2, α, M, λm, Km])
            @printf("  N=%d α=%.2f: M=%d, λ_Mill=%.3f, ⟨K⟩_exact=%.3f\n",
                    N2, α, M, λm, Km)
        end
    end
    mat = reduce(hcat, rows)'
    open(CSV_PANEL2, "w") do io
        println(io, "N,alpha,M,lambda_mill,K_mean")
        writedlm(io, mat, ',')
    end
    println("  → Saved $CSV_PANEL2")
end

function compute_panel3(; N3=50, n_bp=10000)
    println("\n══ Computing Panel 3: Branching process ══")
    α_bp = collect(0.20:0.005:0.42)
    max_gens = [1, 3, 10, 50]

    # Matrix: α, surv_1, surv_3, surv_10, surv_50
    mat = zeros(length(α_bp), 1 + length(max_gens))
    mat[:, 1] = α_bp

    for (gi, max_gen) in enumerate(max_gens)
        for (ai, α) in enumerate(α_bp)
            λ = λ_anal(α, N3)
            ns = 0
            for _ in 1:n_bp
                alive = 1
                survived = false
                for g in 1:max_gen
                    new = 0
                    for _ in 1:min(alive, 500)
                        new += rand_poisson(λ)
                    end
                    alive = new
                    alive == 0 && break
                    if alive > 5000
                        survived = true
                        break
                    end
                end
                (survived || alive > 0) && (ns += 1)
            end
            mat[ai, 1 + gi] = ns / n_bp
        end
        @printf("  depth ≤ %d: done\n", max_gen)
    end

    open(CSV_PANEL3, "w") do io
        println(io, "alpha,surv_depth1,surv_depth3,surv_depth10,surv_depth50")
        writedlm(io, mat, ',')
    end
    println("  → Saved $CSV_PANEL3")
end

function compute_panel4(; N4=25, n_bfs=200)
    println("\n══ Computing Panel 4: BFS depth analysis ══")
    ac4 = α_c_perc(N4)
    α_bfs = collect(round(ac4 - 0.04, digits=2):0.01:min(round(ac4 + 0.10, digits=2), 0.50))

    rows = Vector{Vector{Float64}}()
    for α in α_bfs
        M = round(Int, exp(N4 * α))
        sqrtN = sqrt(Float64(N4))
        d1s = Float64[]
        d2s = Float64[]

        for _ in 1:n_bfs
            Z = randn(N4, M)
            for j in 1:M
                v = view(Z, :, j)
                v .*= sqrtN / norm(v)
            end

            target = Z[:, 1]
            q_tgt = (Z' * target) / N4
            d1_mask = falses(M)
            for i in 2:M
                q_tgt[i] > φ_c && (d1_mask[i] = true)
            end
            d1_idx = findall(d1_mask)
            K1 = length(d1_idx)

            d2_new = 0
            if K1 > 0 && K1 ≤ 200
                seen = copy(d1_mask)
                seen[1] = true
                for idx in d1_idx
                    q_i = (Z' * Z[:, idx]) / N4
                    for j in 1:M
                        if !seen[j] && q_i[j] > φ_c
                            d2_new += 1
                            seen[j] = true
                        end
                    end
                end
            end

            push!(d1s, K1)
            push!(d2s, d2_new)
        end

        push!(rows, [α, M, mean(d1s), mean(d2s)])
        @printf("  α=%.2f: M=%6d, ⟨K₁⟩=%6.2f, ⟨K₂_new⟩=%.4f\n",
                α, M, mean(d1s), mean(d2s))
    end

    mat = reduce(hcat, rows)'
    open(CSV_PANEL4, "w") do io
        println(io, "alpha,M,K1_mean,K2_mean")
        writedlm(io, mat, ',')
    end
    println("  → Saved $CSV_PANEL4")
end

# ──────────────── Run computation if needed ────────────────

need2 = REFRESH_CSV || !isfile(CSV_PANEL2)
need3 = REFRESH_CSV || !isfile(CSV_PANEL3)
need4 = REFRESH_CSV || !isfile(CSV_PANEL4)

need2 && compute_panel2()
need3 && compute_panel3()
need4 && compute_panel4()

if !need2 && !need3 && !need4
    println("\n  All CSV files found — plotting from cache.")
    println("  Use --refresh_csv to recompute.")
end

# ──────────────── Read CSV data ────────────────

d2 = readdlm(CSV_PANEL2, ','; header=true)[1]
d3 = readdlm(CSV_PANEL3, ','; header=true)[1]
d4 = readdlm(CSV_PANEL4, ','; header=true)[1]

# ══════════════════════════════════════════════════════════
# Panel 1: λ(α) — purely analytical
# ══════════════════════════════════════════════════════════
αr = 0.10:0.002:0.55
p1 = plot(xlabel="α", ylabel="λ", title="Expected neighbors λ(α)",
    yscale=:log10, ylims=(1e-3, 1e4), legend=:bottomright, legendfontsize=7)

for N in [30, 50, 75, 150]
    ac = α_c_perc(N)
    λs = [λ_anal(α, N) for α in αr]
    plot!(p1, collect(αr), λs, lw=2, label=@sprintf("N=%d (α_c=%.3f)", N, ac))
end

hline!(p1, [1.0], color=:black, ls=:dash, lw=2, label="λ=1")
vline!(p1, [α_∞], color=:gray, ls=:dot, lw=1.5, label="α=0.25")

# ══════════════════════════════════════════════════════════
# Panel 2: ⟨K⟩ vs α from CSV
# ══════════════════════════════════════════════════════════
p2 = plot(xlabel="α", ylabel="⟨K⟩ (mean neighbor count)",
    title="Neighbors: Gaussian approx vs exact sphere",
    legend=:topleft, legendfontsize=7, yscale=:log10, ylims=(1e-3, 1e2))

α_full = collect(0.24:0.01:0.50)

for N2 in [25, 50]
    corr_factor = exp(N2 * CORR_COEFF)

    # Analytical λ curve (full range)
    λ_full = [λ_anal(α, N2) for α in α_full]
    plot!(p2, α_full, λ_full, lw=2, ls=:dash,
        label=@sprintf("λ Mill's ratio (N=%d)", N2))

    # Empirical data from CSV
    mask = d2[:, 1] .== N2
    α_emp = d2[mask, 2]
    K_means = d2[mask, 5]

    # Filled circles: raw empirical ⟨K⟩
    scatter!(p2, α_emp, max.(K_means, 5e-4), ms=6, markershape=:circle,
        label=@sprintf("⟨K⟩ exact sphere (N=%d)", N2))
    # Grab the auto-assigned fill color of the series just plotted
    fill_color = p2.series_list[end][:seriescolor]
    # Open circles: ⟨K⟩ × e^{N×0.083} (higher-order correction)
    K_corrected = K_means .* corr_factor
    scatter!(p2, α_emp, max.(K_corrected, 5e-4), ms=7,
        markershape=:circle, markercolor=:white, markerstrokewidth=2,
        markerstrokecolor=fill_color,
        label=@sprintf("⟨K⟩×e^{%.1fN} (N=%d)", CORR_COEFF, N2))
end

hline!(p2, [1.0], color=:black, ls=:dash, lw=1.5, label="⟨K⟩=1 (percolation)")

# ══════════════════════════════════════════════════════════
# Panel 3: Branching process from CSV
# ══════════════════════════════════════════════════════════
N3 = 50
ac3 = α_c_perc(N3)
α_bp = d3[:, 1]

p3 = plot(xlabel="α", ylabel="P(cluster survives)",
    title=@sprintf("Branching process (N=%d)", N3),
    legend=:topleft, legendfontsize=7)

for (gi, (max_gen, col)) in enumerate(zip([1, 3, 10, 50], 2:5))
    plot!(p3, α_bp, d3[:, col], lw=2, label="depth ≤ $max_gen")
end

# Theory: P(K≥1) = 1-exp(-λ)
λ_curve = [λ_anal(α, N3) for α in α_bp]
plot!(p3, α_bp, 1 .- exp.(-λ_curve), lw=1.5, ls=:dash,
    color=:black, label="1-e^{-λ} (depth=1 theory)")
vline!(p3, [ac3], color=:gray, ls=:dot, lw=1.5,
    label=@sprintf("α_c=%.3f", ac3))

# ══════════════════════════════════════════════════════════
# Panel 4: BFS depth from CSV
# ══════════════════════════════════════════════════════════
N4 = 25
ac4 = α_c_perc(N4)
α_bfs = d4[:, 1]
d1_mean = d4[:, 3]
d2_mean = d4[:, 4]

p4 = plot(xlabel="α", ylabel="Mean count",
    title=@sprintf("BFS depth (N=%d)", N4),
    legend=:topleft, legendfontsize=7)
plot!(p4, α_bfs, d1_mean, lw=2, marker=:circle, ms=4, label="Depth-1 ⟨K₁⟩")
plot!(p4, α_bfs, d2_mean, lw=2, marker=:square, ms=4, label="Depth-2 new ⟨K₂⟩")
λ_theory = [λ_anal(α, N4) for α in α_bfs]
plot!(p4, α_bfs, λ_theory, lw=1.5, ls=:dash, color=:black, label="λ (theory)")
vline!(p4, [ac4], color=:gray, ls=:dot, lw=1.5,
    label=@sprintf("α_c=%.3f", ac4))

# ══════════════════════════════════════════════════════════
# Combine
# ══════════════════════════════════════════════════════════
p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1000), dpi=150,
    plot_title="LSR Basin Percolation (q_th = φ_c = 1/√2)",
    margin=5Plots.mm)

savefig(p, "percolation_LSR.png")
savefig(p, "percolation_LSR.pdf")
println("\n✓ Saved: percolation_LSR.{png,pdf}")

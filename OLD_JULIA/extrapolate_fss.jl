#=
Finite-Size Scaling Extrapolation
Reads φ(α,T) from 3 P-scales, fits φ vs 1/N linearly, extrapolates to N→∞.
Input:  lse_alpha_sweep_table.csv, lse_s2.csv, lse_s3.csv  (and lsr_*)
Output: lse_extrap.csv, lsr_extrap.csv
=#

using CSV
using DataFrames
using Statistics
using Printf

# ──────────────── Scale definitions (must match GPU scripts) ────────────────
struct Scale
    min_pat::Int
    max_pat::Int
    csv::String
end

lse_scales = [
    Scale(200,   5000,   "lse_alpha_sweep_table.csv"),
    Scale(2000,  50000,  "lse_s2.csv"),
    Scale(10000, 300000, "lse_s3.csv"),
]

lsr_scales = [
    Scale(200,   5000,   "lsr_alpha_sweep_table.csv"),
    Scale(2000,  50000,  "lsr_s2.csv"),
    Scale(10000, 300000, "lsr_s3.csv"),
]

# Compute N for given α and scale parameters
function compute_N(alpha, min_pat, max_pat, alpha_min, alpha_max)
    slope = (max_pat - min_pat) / (alpha_max - alpha_min)
    P = min_pat + slope * (alpha - alpha_min)
    return max(round(Int, log(P) / alpha), 2)
end

# Linear extrapolation: fit φ = a + b/N, return a (the N→∞ value)
function extrapolate(inv_Ns::Vector{Float64}, phis::Vector{Float64})
    n = length(inv_Ns)
    if n == 1
        return phis[1]
    end
    # Least-squares fit: φ = a + b * (1/N)
    x̄ = mean(inv_Ns)
    ȳ = mean(phis)
    Sxx = sum((inv_Ns .- x̄).^2)
    Sxy = sum((inv_Ns .- x̄) .* (phis .- ȳ))
    if Sxx < 1e-20
        return ȳ
    end
    b = Sxy / Sxx
    a = ȳ - b * x̄
    return clamp(a, -0.05, 1.05)  # allow slight overshoot
end

function process_kernel(scales::Vector{Scale}, output_csv::String, kernel_name::String)
    println("Processing $kernel_name...")

    # Read all scale CSVs
    dfs = DataFrame[]
    for sc in scales
        if !isfile(sc.csv)
            println("  WARNING: $(sc.csv) not found, skipping")
            continue
        end
        push!(dfs, CSV.read(sc.csv, DataFrame))
        println("  Read $(sc.csv)")
    end

    if isempty(dfs)
        println("  No data files found!")
        return
    end

    # Extract grid
    alpha_values = dfs[1][:, 1]
    T_columns = names(dfs[1])[2:end]
    T_values = [parse(Float64, replace(col, "T" => "")) for col in T_columns]
    n_alpha = length(alpha_values)
    n_T = length(T_values)
    alpha_min = minimum(alpha_values)
    alpha_max = maximum(alpha_values)

    # Read φ matrices
    phi_matrices = [Matrix(df[:, 2:end]) for df in dfs]
    available_scales = [scales[i] for i in 1:length(dfs)]

    # Extrapolate each (α, T) point
    phi_extrap = zeros(Float64, n_alpha, n_T)

    for i in 1:n_alpha
        alpha = alpha_values[i]
        # Compute N at each scale
        Ns = [compute_N(Float64(alpha), sc.min_pat, sc.max_pat, Float64(alpha_min), Float64(alpha_max))
              for sc in available_scales]
        inv_Ns = 1.0 ./ Ns

        for j in 1:n_T
            phis = [phi_matrices[k][i, j] for k in 1:length(dfs)]
            phi_extrap[i, j] = extrapolate(inv_Ns, phis)
        end
    end

    # Save extrapolated CSV
    open(output_csv, "w") do f
        write(f, "alpha")
        for T in T_values; write(f, @sprintf(",T%.2f", T)); end
        write(f, "\n")
        for i in 1:n_alpha
            write(f, @sprintf("%.2f", alpha_values[i]))
            for j in 1:n_T; write(f, @sprintf(",%.4f", phi_extrap[i, j])); end
            write(f, "\n")
        end
    end
    println("  Saved: $output_csv")

    # Print N values and sample extrapolations
    println("\n  N values per scale:")
    for idx in [1, n_alpha÷4, n_alpha÷2, 3*n_alpha÷4, n_alpha]
        alpha = alpha_values[idx]
        Ns = [compute_N(Float64(alpha), sc.min_pat, sc.max_pat, Float64(alpha_min), Float64(alpha_max))
              for sc in available_scales]
        j_1 = findfirst(t -> abs(t - 1.0) < 0.01, T_values)
        phis = [phi_matrices[k][idx, j_1] for k in 1:length(dfs)]
        @printf("    α=%.2f: N=%s, φ(T=1)=%s → extrap=%.4f\n",
                alpha, string(Ns), string(round.(phis, digits=4)), phi_extrap[idx, j_1])
    end
    println()
end

# ──────────────── Main ────────────────
println("=" ^ 70)
println("Finite-Size Scaling Extrapolation")
println("=" ^ 70)

process_kernel(lse_scales, "lse_extrap.csv", "LSE")
process_kernel(lsr_scales, "lsr_extrap.csv", "LSR")

println("=" ^ 70)
println("Done. Use maps.jl to plot.")
println("=" ^ 70)

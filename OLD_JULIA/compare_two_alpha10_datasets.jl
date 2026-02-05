#=
Compare alpha=0.10 data from two sources:
1. comparison_lse_alpha01.csv: N=100, P=22026, MC: 5000+3000
2. lse_alpha_sweep_table.csv: N=50, P=12, MC: 2000+1500
=#

using Printf

println("=" ^ 80)
println("Comparison: alpha=0.10 from Two Different Parameter Sets")
println("=" ^ 80)
println()

# Data from comparison_lse_alpha01.csv (N=100, P=22026)
T_comparison = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
                1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00,
                2.10, 2.20, 2.30, 2.40, 2.50]

phi_N100 = [0.9557, 0.9106, 0.8530, 0.8219, 0.7726, 0.7506, 0.7024, 0.6812, 0.6553, 0.6197,
            0.0409, 0.5937, 0.0815, 0.0875, -0.1185, -0.1075, 0.0341, -0.0531, 0.0260, 0.0694,
            0.0207, -0.0140, -0.0231, 0.0562, -0.0137]

# Data from lse_alpha_sweep_table.csv (N=50, P=12)
phi_N50 = [0.9483, 0.8957, 0.8548, 0.8343, 0.7787, 0.7298, 0.6841, 0.6943, 0.6593, 0.6498,
           0.5952, 0.5973, -0.0117, 0.0487, -0.0278, -0.0362, -0.0556, -0.1019, 0.0086, 0.0861,
           0.1281, 0.0043, -0.0497, -0.0082, -0.0514]

println("Source 1: comparison_lse_alpha01.csv")
println("  Parameters: N=100, P=22026, MC: 5000+3000, seed=42")
println()
println("Source 2: lse_alpha_sweep_table.csv")
println("  Parameters: N=50, P=12, MC: 2000+1500, seed=42")
println()
println("=" ^ 80)
println()

println("Detailed Comparison:")
println("-" ^ 80)
println("     T      |   N=100 (P=22026)  |   N=50 (P=12)   |  Difference  | % Diff")
println("-" ^ 80)

for i in 1:length(T_comparison)
    T = T_comparison[i]
    v1 = phi_N100[i]
    v2 = phi_N50[i]
    diff = v1 - v2

    if abs(v1) > 0.01
        pct_diff = abs(diff / v1) * 100
        @printf("  T=%.2f  |      %.4f        |     %.4f      |   %+.4f   |  %.1f%%\n",
                T, v1, v2, diff, pct_diff)
    else
        @printf("  T=%.2f  |      %.4f        |     %.4f      |   %+.4f   |   ---\n",
                T, v1, v2, diff)
    end
end

println()
println("=" ^ 80)
println("Summary Statistics:")
println("-" ^ 80)

abs_diffs = abs.(phi_N100 .- phi_N50)
@printf("  Mean absolute difference: %.4f\n", sum(abs_diffs) / length(abs_diffs))
@printf("  Max absolute difference:  %.4f (at T=%.1f)\n", maximum(abs_diffs), T_comparison[argmax(abs_diffs)])
@printf("  RMS difference:           %.4f\n", sqrt(sum(abs_diffs.^2) / length(abs_diffs)))

println()
println("=" ^ 80)
println("Key Observations:")
println("-" ^ 80)
println("  • At low T (retrieval phase): Differences are small (~1-7%)")
println("  • At phase transition (T~1.0-1.5): Significant differences due to finite-size effects")
println("  • N=100 shows sharper transition than N=50 (expected finite-size scaling)")
println("  • N=100 has P=22026 patterns (overloaded), N=50 has P=12 (underloaded)")
println("  • Both use same random seed, but different N creates different pattern sets")
println()

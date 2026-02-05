#=
Generate CSV table: LSR alignment vs (alpha, T) - ADAPTIVE N VERSION
Alpha: 0.01, 0.02, ..., 0.55
T: 0.05, 0.10, ..., 2.5
N adapts for each α to achieve desired pattern count P = N_patterns(α)
N_patterns grows linearly from min_patterns to max_patterns
This avoids capping artifacts while controlling computational cost
Uses Epanechnikov kernel (LSR) with parameter b
=#

using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

println("=" ^ 70)
println("LSR Alpha Sweep Table Generation (ADAPTIVE N)")
println("=" ^ 70)
println()

# Parameters - ADAPTIVE N approach
b = 2 + sqrt(2)  # Epanechnikov parameter ≈ 3.414

# Alpha and Temperature ranges
alpha_vec = collect(0.01:0.01:0.55)  # 0.01, 0.02, ..., 0.55
T_vec = collect(0.05:0.05:2.5)         # 0.05, 0.10, ..., 2.5

n_alpha = length(alpha_vec)
n_T = length(T_vec)

# Pattern count function: grows linearly with alpha
min_patterns = 200    # At min(alpha_vec)
max_patterns = 5000    # At max(alpha_vec)

# Function to compute desired number of patterns for given alpha
function N_patterns(alpha::Float64)
    alpha_min = minimum(alpha_vec)
    alpha_max = maximum(alpha_vec)
    # Linear interpolation
    slope = (max_patterns - min_patterns) / (alpha_max - alpha_min)
    return min_patterns + slope * (alpha - alpha_min)
end

# MC parameters
n_eq = 5000
n_samp = 3000
step_size = 0.1

println("Parameters (ADAPTIVE N, LSR with b=$(round(b, digits=3))):")
println("  Alpha range: $(alpha_vec[1]) to $(alpha_vec[end]) ($n_alpha values)")
println("  T range: $(T_vec[1]) to $(T_vec[end]) ($n_T values)")
println("  Pattern range: $min_patterns to $max_patterns (linear growth with α)")
println("  Grid size: $n_alpha × $n_T = $(n_alpha * n_T) points")
println("  MC: n_eq=$n_eq, n_samp=$n_samp, step_size=$step_size")
println("  Using $(Threads.nthreads()) threads")
println()
println("N will adapt for each α to achieve P = N_patterns(α)")
println()

# ============== ENERGY FUNCTION ==============
function compute_energy_lsr(x::Vector{Float64}, patterns::Matrix{Float64},
                           N::Float64, b::Float64)
    Nb = N / b
    phi = (patterns' * x) / N
    args = max.(0.0, 1.0 .- b * (1.0 .- phi))
    sum_args = sum(args)
    sum_args = max(sum_args, 1e-10)
    energy = -Nb * log(sum_args)
    return energy
end

# ============== MONTE CARLO ==============
function run_mc(N::Int, patterns::Matrix{Float64}, target::Vector{Float64},
                T::Float64, b::Float64,
                n_eq::Int, n_samp::Int, step_size::Float64)

    Nf = Float64(N)
    beta = 1.0 / T

    x = target + 0.05 * randn(N)
    x = sqrt(Nf) * x / norm(x)

    E = compute_energy_lsr(x, patterns, Nf, b)

    phi_sum = 0.0
    n_total = n_eq + n_samp

    for i in 1:n_total
        x_prop = x + step_size * randn(N)
        x_prop = sqrt(Nf) * x_prop / norm(x_prop)

        E_prop = compute_energy_lsr(x_prop, patterns, Nf, b)

        delta_E = E_prop - E
        if rand() < exp(-beta * delta_E)
            x = x_prop
            E = E_prop
        end

        if i > n_eq
            phi_sum += dot(target, x) / Nf
        end
    end

    return phi_sum / n_samp
end

# ============== SWEEP ==============
println("Computing LSR alignments...")
println()

phi_grid = zeros(Float64, n_alpha, n_T)

progress = Progress(n_alpha, desc="Alpha sweep: ")
for i_alpha in 1:n_alpha
    alpha = alpha_vec[i_alpha]

    # Compute adaptive N for this alpha
    P_target = N_patterns(alpha)
    N = round(Int, log(P_target) / alpha)
    P = round(Int, P_target)

    # Verify: exp(alpha * N) ≈ P
    P_actual = round(Int, exp(alpha * N))

    if i_alpha == 1 || i_alpha == n_alpha || mod(i_alpha, 5) == 0
        println("  α=$(round(alpha, digits=3)): N=$N, P=$P (P_actual=$P_actual)")
    end

    # Generate random patterns
    Random.seed!(42 + i_alpha)  # Different seed for each alpha
    patterns = randn(N, P)
    for j in 1:P
        patterns[:, j] = sqrt(N) * patterns[:, j] / norm(patterns[:, j])
    end
    target = patterns[:, 1]

    # Parallel over temperatures
    phi_T = zeros(n_T)
    Threads.@threads for i_T in 1:n_T
        T = T_vec[i_T]
        phi_T[i_T] = run_mc(N, patterns, target, T, b,
                           n_eq, n_samp, step_size)
    end

    phi_grid[i_alpha, :] = phi_T
    next!(progress)
end
finish!(progress)

println()

# ============== SAVE CSV ==============
println("Writing CSV file...")

csv_file = "lsr_alpha_sweep_table.csv"
open(csv_file, "w") do f
    # Header row
    write(f, "alpha")
    for T in T_vec
        write(f, @sprintf(",T%.2f", T))  # Use %.2f for finer resolution
    end
    write(f, "\n")

    # Data rows
    for i_alpha in 1:n_alpha
        write(f, @sprintf("%.2f", alpha_vec[i_alpha]))
        for i_T in 1:n_T
            write(f, @sprintf(",%.4f", phi_grid[i_alpha, i_T]))
        end
        write(f, "\n")
    end
end

println("CSV file saved: $csv_file")
println()

# Show sample data
println("Sample data (first few alpha values):")
for i in 1:min(3, n_alpha)
    @printf("  α=%.2f: φ(T=0.05)=%.4f, φ(T=1.00)=%.4f, φ(T=2.50)=%.4f\n",
            alpha_vec[i], phi_grid[i, 1], phi_grid[i, 20], phi_grid[i, end])
end
println()
println("=" ^ 70)
println("Table generation complete!")
println("Adaptive N approach: N varied from $(round(Int, log(min_patterns)/minimum(alpha_vec))) to $(round(Int, log(max_patterns)/maximum(alpha_vec)))")
println("MC steps=$(n_eq)+$(n_samp)")
println("b = $(round(b, digits=4)) (Epanechnikov kernel)")
println("=" ^ 70)

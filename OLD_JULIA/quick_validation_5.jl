#=
Quick Validation: Key Predictions from the Paper
CPU-only multithreaded version (Float64 precision)
Julia translation of quick_validation_5.m
=#

using CairoMakie
using Random
using Statistics
using LinearAlgebra
using Printf

println("=== Quick Validation of DAM Theory ===\n")

# Parameters matching MATLAB
b = 2 + sqrt(2)  # ≈ 3.4142
N = 100
alpha = 0.1
P = round(Int, exp(alpha * N))  # ≈ 148 patterns

println("Parameters:")
println("  N = $N")
println("  α = $alpha")
println("  P = $P patterns")
println("  b = $(round(b, digits=4))")
println()

# Generate random patterns
println("Generating patterns...")
Random.seed!(42)
patterns = randn(N, P)
for j in 1:P
    patterns[:, j] = sqrt(N) * patterns[:, j] / norm(patterns[:, j])
end
target = patterns[:, 1]

# Temperature range
T_range = collect(0.0:0.1:2.5)
n_T = length(T_range)

println("Temperature range: $(T_range[1]) to $(T_range[end]) ($(n_T) points)")
println()

# MC parameters (matching MATLAB)
n_eq = 5000
n_samp = 3000
step_size = 0.1
betanet_lse = 1.0

println("MC parameters: n_eq=$n_eq, n_samp=$n_samp, step_size=$step_size")
println("Using $(Threads.nthreads()) threads")
println()

#= ============== ENERGY FUNCTIONS ============== =#

function compute_energy_lse(x::Vector{Float64}, patterns::Matrix{Float64},
                           N::Float64, betanet::Float64)
    # Compute alignments
    phi = (patterns' * x) / N  # [P]

    # LSE: E = -(1/βnet) * ln(Σ exp(-βnet*N*(1-φ)))
    log_args = -betanet * N * (1.0 .- phi)
    max_arg = maximum(log_args)
    energy = -(max_arg + log(sum(exp.(log_args .- max_arg)))) / betanet

    return energy
end

function compute_energy_lsr(x::Vector{Float64}, patterns::Matrix{Float64},
                           N::Float64, b::Float64)
    Nb = N / b

    # Compute alignments
    phi = (patterns' * x) / N  # [P]

    # LSR: E = -(N/b) * ln(Σ max(0, 1-b*(1-φ)))
    args = max.(0.0, 1.0 .- b * (1.0 .- phi))
    sum_args = sum(args)
    sum_args = max(sum_args, 1e-10)  # Avoid log(0)

    energy = -Nb * log(sum_args)

    return energy
end

#= ============== MONTE CARLO SIMULATION ============== =#

"""
Run Monte Carlo simulation for a single (T, kernel) combination.
Returns average alignment φ.
"""
function run_mc(N::Int, patterns::Matrix{Float64}, target::Vector{Float64},
                T::Float64, kernel::Symbol, betanet_lse::Float64, b::Float64,
                n_eq::Int, n_samp::Int, step_size::Float64)

    Nf = Float64(N)
    beta = 1.0 / T

    # Initialize state near target (5% noise - matches MATLAB)
    x = target + 0.05 * randn(N)
    x = sqrt(Nf) * x / norm(x)

    # Compute initial energy
    E = if kernel == :LSE
        compute_energy_lse(x, patterns, Nf, betanet_lse)
    else
        compute_energy_lsr(x, patterns, Nf, b)
    end

    # Combined equilibration and sampling
    phi_sum = 0.0
    n_total = n_eq + n_samp

    for i in 1:n_total
        # Propose new state
        x_prop = x + step_size * randn(N)
        x_prop = sqrt(Nf) * x_prop / norm(x_prop)

        # Compute proposed energy
        E_prop = if kernel == :LSE
            compute_energy_lse(x_prop, patterns, Nf, betanet_lse)
        else
            compute_energy_lsr(x_prop, patterns, Nf, b)
        end

        # Accept/reject
        delta_E = E_prop - E
        if rand() < exp(-beta * delta_E)
            x = x_prop
            E = E_prop
        end

        # Accumulate alignment during sampling phase
        if i > n_eq
            phi_sum += dot(target, x) / Nf
        end
    end

    return phi_sum / n_samp
end

#= ============== RUN SIMULATIONS ============== =#

println("Running MC simulations (multithreaded CPU)...")
println()

# Preallocate results
phi_LSE = zeros(n_T)
phi_LSR = zeros(n_T)

# Run all temperatures in parallel using multithreading
Threads.@threads for i in 1:n_T
    T = T_range[i]

    # Skip T=0 to avoid division by zero
    if T < 1e-6
        phi_LSE[i] = 1.0
        phi_LSR[i] = 1.0
    else
        phi_LSE[i] = run_mc(N, patterns, target, T, :LSE,
                           betanet_lse, b, n_eq, n_samp, step_size)
        phi_LSR[i] = run_mc(N, patterns, target, T, :LSR,
                           betanet_lse, b, n_eq, n_samp, step_size)
    end

    @printf("  T=%.1f: LSE=%.3f, LSR=%.3f\n", T, phi_LSE[i], phi_LSR[i])
end

println()

#= ============== THEORETICAL PREDICTIONS ============== =#

# Theory curves
T_fine = range(0.0, 2.5, length=100)

# LSE theory: φ = (1/2)(-T + √(T² + 4))
phi_theory_LSE = @. 0.5 * (-T_fine + sqrt(T_fine^2 + 4))

# LSR theory: solve (bT+1)y² - (2+T+Tb)y + T = 0, where y = 1-φ
function theoretical_alignment_LSR(T::Float64, b::Float64)
    A = b*T + 1
    B = -(2 + T + T*b)
    C = T

    discriminant = B^2 - 4*A*C
    discriminant < 0 && return NaN

    y1 = (-B - sqrt(discriminant)) / (2*A)
    y2 = (-B + sqrt(discriminant)) / (2*A)

    # Choose solution with y in [0, 1]
    y = (0 <= y1 <= 1) ? y1 : y2
    (y < 0 || y > 1) && return NaN

    phi = 1 - y

    # Check support constraint: phi > 1 - 1/b
    phi <= 1 - 1/b && return NaN

    return phi
end

phi_theory_LSR = [theoretical_alignment_LSR(T, b) for T in T_fine]

# Free energy
f_ret_LSE = zeros(length(T_fine))
for (i, T) in enumerate(T_fine)
    phi = phi_theory_LSE[i]
    if phi < 1 - 1e-10
        f_ret_LSE[i] = (1 - phi) - (T/2) * log(1 - phi^2)
    else
        f_ret_LSE[i] = 0
    end
end

f_ret_LSR = zeros(length(T_fine))
for (i, T) in enumerate(T_fine)
    phi = phi_theory_LSR[i]
    if !isnan(phi) && phi > 1 - 1/b + 1e-10 && phi < 1 - 1e-10
        u = -log(1 - b*(1-phi)) / b
        s = 0.5 * log(1 - phi^2)
        f_ret_LSR[i] = u - T*s
    else
        f_ret_LSR[i] = NaN
    end
end

#= ============== PLOTTING ============== =#

println("Generating figure...")

fig = Figure(size=(800, 350), backgroundcolor=:white)

# Left panel: Alignment vs Temperature
ax1 = Axis(fig[1, 1],
           xlabel="Temperature T",
           ylabel="Alignment φ",
           xgridvisible=true,
           ygridvisible=true)
limits!(ax1, 0, 2.5, 0, 1)

# Theory curves
lines!(ax1, T_fine, phi_theory_LSE, color=:black, linewidth=2, label="LSE theory")
lines!(ax1, T_fine, phi_theory_LSR, color=:blue, linewidth=2, label="LSR theory")

# MC results
scatter!(ax1, T_range, phi_LSE, color=:black, marker=:circle,
         markersize=10, label="LSE MC")
scatter!(ax1, T_range, phi_LSR, color=:blue, marker=:rect,
         markersize=10, label="LSR MC")

axislegend(ax1, position=:lb, framevisible=true, fontsize=13)

# Right panel: Free Energy
ax2 = Axis(fig[1, 2],
           xlabel="Temperature T",
           ylabel="Free energy f_ret",
           xgridvisible=true,
           ygridvisible=true)
limits!(ax2, 0, 2.5, 0, 1.5)

lines!(ax2, T_fine, f_ret_LSE, color=:black, linewidth=2, label="LSE")
lines!(ax2, T_fine, f_ret_LSR, color=:blue, linewidth=2, label="LSR (b=3.41)")

axislegend(ax2, position=:lt, framevisible=true, fontsize=13)

# Save
save("quick_validation.png", fig, px_per_unit=2)
save("quick_validation.eps", fig)

println("\nResults saved to quick_validation.png and quick_validation.eps")
println("\n=== Validation Complete ===")

display(fig)

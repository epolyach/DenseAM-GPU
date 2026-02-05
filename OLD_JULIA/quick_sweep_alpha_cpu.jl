#=
Quick Alpha Sweep - CPU Multithreaded Phase Diagrams
EXACTLY matches quick_validation_5.jl structure, just looped over alpha
Float64 precision, CPU-only
=#

using CairoMakie
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

println("=" ^ 60)
println("Phase Diagrams - CPU Multithreaded (quick_validation style)")
println("=" ^ 60)
println()

# Parameters
b = 2 + sqrt(2)  # ≈ 3.4142
N = 50

# Grid resolution
n_alpha = 40
n_T = 40
alpha_min = 0.02
alpha_max = 0.55
T_min = 0.05
T_max = 2.5

alpha_vec = collect(range(alpha_min, alpha_max, length=n_alpha))
T_vec = collect(range(T_min, T_max, length=n_T))

# MC parameters (matching quick_validation_5.jl)
n_eq = 2000
n_samp = 1500
step_size = 0.1
betanet_lse = 1.0
max_patterns = 50000

println("Parameters:")
println("  N = $N")
println("  Grid: $n_alpha × $n_T = $(n_alpha * n_T) points")
println("  Alpha range: $alpha_min to $alpha_max")
println("  T range: $T_min to $T_max")
println("  b = $(round(b, digits=4))")
println("  MC: n_eq=$n_eq, n_samp=$n_samp, step_size=$step_size")
println("  Using $(Threads.nthreads()) threads")
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
EXACTLY as in quick_validation_5.jl
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

#= ============== SWEEP OVER ALPHA ============== =#

println("Computing phase diagrams (sweeping over alpha)...")
println()

phi_grid_LSE = zeros(Float64, n_alpha, n_T)
phi_grid_LSR = zeros(Float64, n_alpha, n_T)

# Loop over alpha (EXACTLY like running quick_validation_5.jl for each alpha)
progress = Progress(n_alpha, desc="Alpha sweep ")
for i_alpha in 1:n_alpha
    alpha = alpha_vec[i_alpha]
    P = min(round(Int, exp(alpha * N)), max_patterns)

    # Generate random patterns (EXACTLY like quick_validation_5.jl)
    Random.seed!(42)  # Same seed for all alphas to match quick_validation pattern
    patterns = randn(N, P)
    for j in 1:P
        patterns[:, j] = sqrt(N) * patterns[:, j] / norm(patterns[:, j])
    end
    target = patterns[:, 1]

    # Preallocate results for this alpha
    phi_LSE = zeros(n_T)
    phi_LSR = zeros(n_T)

    # Run all temperatures in parallel using multithreading
    # EXACTLY like quick_validation_5.jl lines 152-167
    Threads.@threads for i in 1:n_T
        T = T_vec[i]

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
    end

    # Store results for this alpha
    phi_grid_LSE[i_alpha, :] = phi_LSE
    phi_grid_LSR[i_alpha, :] = phi_LSR

    next!(progress)
end
finish!(progress)

println()

#= ============== THEORETICAL PREDICTIONS ============== =#

function critical_alpha_lse(T::Real)
    phi = 0.5 * (-T + sqrt(T^2 + 4))
    f_ret = (1 - phi) - (T/2) * log(1 - phi^2)
    alpha_c = 0.5 * (1 - f_ret)^2
    return clamp(alpha_c, 0.0, 0.5)
end

function critical_alpha_lsr(T::Real, b::Real)
    A = b*T + 1
    B = -(2 + T + T*b)
    C = T
    disc = B^2 - 4*A*C

    disc < 0 && return NaN

    y = (-B - sqrt(disc)) / (2*A)
    phi = 1 - y

    (phi <= 1 - 1/b || phi > 1 || phi < 0) && return NaN

    u = -(1/b) * log(1 - b*(1-phi))
    s = 0.5 * log(1 - phi^2)
    f_ret = u - T * s

    alpha_c = 0.5 * (1 - f_ret)^2
    return clamp(alpha_c, 0.0, 0.5)
end

function find_T_max_lsr(b::Real, alpha_th::Real)
    for T in range(0.01, 3.0, length=1000)
        ac = critical_alpha_lsr(T, b)
        !isnan(ac) && ac <= alpha_th && return T
    end
    return NaN
end

#= ============== PLOTTING ============== =#

println("Generating figure...")

T_theory = range(0.001, 2.5, length=500)
alpha_c_lse = [critical_alpha_lse(T) for T in T_theory]
alpha_c_lsr = [critical_alpha_lsr(T, b) for T in T_theory]

alpha_th = 0.5 * (1 - 1/b)^2
T_max = find_T_max_lsr(b, alpha_th)

fig = Figure(size=(1400, 600), backgroundcolor=:white)
cmap = cgrad([:darkblue, :blue, :white, :red, :darkred], rev=true)

# LSE Panel
ax1 = Axis(fig[1, 1],
           xlabel=L"\alpha = \ln(P)/N",
           ylabel=L"T",
           title="LSE (Gaussian Kernel)",
           titlesize=18)

hm1 = heatmap!(ax1, alpha_vec, T_vec, phi_grid_LSE',
               colormap=cmap, colorrange=(0, 1))
Colorbar(fig[1, 2], hm1, label=L"\phi")

lines!(ax1, alpha_c_lse, collect(T_theory), color=:black, linewidth=2.5)
contour!(ax1, alpha_vec, T_vec, phi_grid_LSE', levels=[0.5],
         color=:white, linewidth=2, linestyle=:dash)
vlines!(ax1, [0.5], color=:black, linewidth=1.5, linestyle=:dash)

text!(ax1, 0.12, 0.4, text="Retrieval", fontsize=14, color=:white)
text!(ax1, 0.35, 1.6, text="Spin-Glass", fontsize=14, color=:black)
text!(ax1, 0.40, 0.12, text=L"\alpha_c(0)=0.5", fontsize=11, color=:white)

# LSR Panel
ax2 = Axis(fig[1, 3],
           xlabel=L"\alpha = \ln(P)/N",
           ylabel=L"T",
           title=@sprintf("LSR (Epanechnikov, b=%.2f)", b),
           titlesize=18)

hm2 = heatmap!(ax2, alpha_vec, T_vec, phi_grid_LSR',
               colormap=cmap, colorrange=(0, 1))
Colorbar(fig[1, 4], hm2, label=L"\phi")

valid_idx = .!isnan.(alpha_c_lsr) .& (alpha_c_lsr .> alpha_th) .& (alpha_c_lsr .<= 0.5)
lines!(ax2, alpha_c_lsr[valid_idx], collect(T_theory)[valid_idx],
       color=:black, linewidth=2.5)

lines!(ax2, [alpha_th, alpha_th], [0, T_max], color=:black, linewidth=2.5)
lines!(ax2, [alpha_th, 0.5], [T_max, T_max], color=:black, linewidth=1.5, linestyle=:dot)

contour!(ax2, alpha_vec, T_vec, phi_grid_LSR', levels=[0.5],
         color=:white, linewidth=2, linestyle=:dash)
vlines!(ax2, [0.5], color=:black, linewidth=1.5, linestyle=:dash)

text!(ax2, 0.08, 1.0, text="Retrieval", fontsize=14, color=:white)
text!(ax2, 0.35, 1.6, text="Spin-Glass", fontsize=14, color=:black)
text!(ax2, 0.40, 0.12, text=L"\alpha_c(0)=0.5", fontsize=11, color=:white)
text!(ax2, alpha_th + 0.01, 0.12, text=@sprintf("α_th=%.2f", alpha_th),
      fontsize=10, color=:white)
text!(ax2, alpha_th - 0.08, T_max + 0.1, text=@sprintf("T_max=%.2f", T_max),
      fontsize=10, color=:black)

Label(fig[0, :], @sprintf("Phase Diagrams (N=%d, CPU-multithreaded, quick_validation style)", N),
      fontsize=20)

save("quick_sweep_alpha_cpu.png", fig, px_per_unit=2)
save("quick_sweep_alpha_cpu.eps", fig)
println("\nFigures saved: quick_sweep_alpha_cpu.{png,eps}")

display(fig)

println("\n=== Phase diagram generation complete ===")

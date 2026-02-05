#=
CPU-Multithreaded Phase Boundary Detection for Dense Associative Memory
Float64 precision with independent pattern realizations for each (α, T) point
=#

using CairoMakie
using Random
using Statistics
using LinearAlgebra
using ProgressMeter
using Printf

struct SimConfig
    N::Int
    n_alpha::Int
    n_T::Int
    alpha_min::Float64
    alpha_max::Float64
    T_min::Float64
    T_max::Float64
    betanet_lse::Float64
    b::Float64
    n_eq::Int
    n_samp::Int
    step_size::Float64
    n_trials::Int
    max_patterns::Int
end

function default_config()
    SimConfig(
        50,                 # N: Network dimension
        40,                 # n_alpha: Grid resolution in α direction
        40,                 # n_T: Grid resolution in temperature direction
        0.02,               # alpha_min
        0.55,               # alpha_max
        0.05,               # T_min
        2.5,                # T_max
        1.0,                # betanet_lse
        2 + sqrt(2),        # b ≈ 3.414
        2000,               # n_eq
        1500,               # n_samp
        0.2,                # step_size
        10,                 # n_trials: Independent pattern realizations
        5000                # max_patterns
    )
end

#= ============== ENERGY FUNCTIONS ============== =#

function compute_energy_lse(x::Vector{Float64}, patterns::Matrix{Float64},
                           N::Float64, betanet::Float64)
    phi = (patterns' * x) / N
    log_args = -betanet * N * (1.0 .- phi)
    max_arg = maximum(log_args)
    energy = -(max_arg + log(sum(exp.(log_args .- max_arg)))) / betanet
    return energy
end

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

#= ============== MONTE CARLO SIMULATION ============== =#

function run_mc_single(N::Int, patterns::Matrix{Float64}, target::Vector{Float64},
                      T::Float64, kernel::Symbol, betanet_lse::Float64, b::Float64,
                      n_eq::Int, n_samp::Int, step_size::Float64, rng::Random.AbstractRNG)

    Nf = Float64(N)
    beta = 1.0 / T

    # Initialize state near target (5% noise)
    x = target + 0.05 * randn(rng, N)
    x = sqrt(Nf) * x / norm(x)

    # Compute initial energy
    E = if kernel == :LSE
        compute_energy_lse(x, patterns, Nf, betanet_lse)
    else
        compute_energy_lsr(x, patterns, Nf, b)
    end

    # Equilibration
    for step in 1:n_eq
        x_prop = x + step_size * randn(rng, N)
        x_prop = sqrt(Nf) * x_prop / norm(x_prop)

        E_prop = if kernel == :LSE
            compute_energy_lse(x_prop, patterns, Nf, betanet_lse)
        else
            compute_energy_lsr(x_prop, patterns, Nf, b)
        end

        delta_E = E_prop - E
        if rand(rng) < exp(-beta * delta_E)
            x = x_prop
            E = E_prop
        end
    end

    # Sampling
    phi_sum = 0.0
    for step in 1:n_samp
        x_prop = x + step_size * randn(rng, N)
        x_prop = sqrt(Nf) * x_prop / norm(x_prop)

        E_prop = if kernel == :LSE
            compute_energy_lse(x_prop, patterns, Nf, betanet_lse)
        else
            compute_energy_lsr(x_prop, patterns, Nf, b)
        end

        delta_E = E_prop - E
        if rand(rng) < exp(-beta * delta_E)
            x = x_prop
            E = E_prop
        end

        phi_sum += dot(target, x) / Nf
    end

    return phi_sum / n_samp
end

#= ============== PARALLEL PHASE DIAGRAM COMPUTATION ============== =#

function compute_phase_diagram_cpu(config::SimConfig, kernel::Symbol)
    alpha_vec = range(config.alpha_min, config.alpha_max, length=config.n_alpha)
    T_vec = range(config.T_min, config.T_max, length=config.n_T)

    phi_grid = zeros(Float64, config.n_alpha, config.n_T)

    # Total grid points
    n_total = config.n_alpha * config.n_T

    println("  Processing $n_total grid points with $(Threads.nthreads()) threads...")
    println("  Each point: $(config.n_trials) independent pattern realizations")

    # Flatten the grid for parallel processing
    grid_points = [(i, j) for i in 1:config.n_alpha for j in 1:config.n_T]

    # Progress tracking
    progress = Progress(n_total, desc="$kernel ")

    # Parallel loop over all grid points
    Threads.@threads for idx in 1:n_total
        i, j = grid_points[idx]
        alpha = alpha_vec[i]
        T = T_vec[j]

        # Skip T=0 to avoid division by zero
        if T < 1e-6
            phi_grid[i, j] = 1.0
            next!(progress)
            continue
        end

        P = min(round(Int, exp(alpha * config.N)), config.max_patterns)

        # Average over multiple independent pattern realizations
        phi_trials = zeros(Float64, config.n_trials)

        for trial in 1:config.n_trials
            # Create thread-safe RNG
            rng = Random.MersenneTwister(hash((alpha, T, trial, kernel)))

            # Generate NEW independent patterns for this trial
            patterns = randn(rng, config.N, P)
            for k in 1:P
                patterns[:, k] = sqrt(config.N) * patterns[:, k] / norm(patterns[:, k])
            end
            target = patterns[:, 1]

            # Run MC simulation
            phi_trials[trial] = run_mc_single(config.N, patterns, target, T, kernel,
                                             config.betanet_lse, config.b,
                                             config.n_eq, config.n_samp,
                                             config.step_size, rng)
        end

        # Average over trials
        phi_grid[i, j] = mean(phi_trials)

        next!(progress)
    end

    finish!(progress)

    return collect(alpha_vec), collect(T_vec), phi_grid
end

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

function plot_phase_diagrams(alpha_lse, T_lse, phi_lse,
                             alpha_lsr, T_lsr, phi_lsr,
                             config::SimConfig)

    T_theory = range(0.001, 2.5, length=500)
    alpha_c_lse = [critical_alpha_lse(T) for T in T_theory]
    alpha_c_lsr = [critical_alpha_lsr(T, config.b) for T in T_theory]

    alpha_th = 0.5 * (1 - 1/config.b)^2
    T_max = find_T_max_lsr(config.b, alpha_th)

    fig = Figure(size=(1400, 600), backgroundcolor=:white)
    cmap = cgrad([:darkblue, :blue, :white, :red, :darkred], rev=true)

    # LSE Panel
    ax1 = Axis(fig[1, 1],
               xlabel=L"\alpha = \ln(P)/N",
               ylabel=L"T",
               title="LSE (Gaussian Kernel)",
               titlesize=18)

    hm1 = heatmap!(ax1, alpha_lse, T_lse, phi_lse',
                   colormap=cmap, colorrange=(0, 1))
    Colorbar(fig[1, 2], hm1, label=L"\phi")

    lines!(ax1, alpha_c_lse, collect(T_theory), color=:black, linewidth=2.5)
    contour!(ax1, alpha_lse, T_lse, phi_lse', levels=[0.5],
             color=:white, linewidth=2, linestyle=:dash)
    vlines!(ax1, [0.5], color=:black, linewidth=1.5, linestyle=:dash)

    text!(ax1, 0.12, 0.4, text="Retrieval", fontsize=14, color=:white)
    text!(ax1, 0.35, 1.6, text="Spin-Glass", fontsize=14, color=:black)
    text!(ax1, 0.40, 0.12, text=L"\alpha_c(0)=0.5", fontsize=11, color=:white)

    # LSR Panel
    ax2 = Axis(fig[1, 3],
               xlabel=L"\alpha = \ln(P)/N",
               ylabel=L"T",
               title=@sprintf("LSR (Epanechnikov, b=%.2f)", config.b),
               titlesize=18)

    hm2 = heatmap!(ax2, alpha_lsr, T_lsr, phi_lsr',
                   colormap=cmap, colorrange=(0, 1))
    Colorbar(fig[1, 4], hm2, label=L"\phi")

    valid_idx = .!isnan.(alpha_c_lsr) .& (alpha_c_lsr .> alpha_th) .& (alpha_c_lsr .<= 0.5)
    lines!(ax2, alpha_c_lsr[valid_idx], collect(T_theory)[valid_idx],
           color=:black, linewidth=2.5)

    lines!(ax2, [alpha_th, alpha_th], [0, T_max], color=:black, linewidth=2.5)
    lines!(ax2, [alpha_th, 0.5], [T_max, T_max], color=:black, linewidth=1.5, linestyle=:dot)

    contour!(ax2, alpha_lsr, T_lsr, phi_lsr', levels=[0.5],
             color=:white, linewidth=2, linestyle=:dash)
    vlines!(ax2, [0.5], color=:black, linewidth=1.5, linestyle=:dash)

    text!(ax2, 0.08, 1.0, text="Retrieval", fontsize=14, color=:white)
    text!(ax2, 0.35, 1.6, text="Spin-Glass", fontsize=14, color=:black)
    text!(ax2, 0.40, 0.12, text=L"\alpha_c(0)=0.5", fontsize=11, color=:white)
    text!(ax2, alpha_th + 0.01, 0.12, text=@sprintf("α_th=%.2f", alpha_th),
          fontsize=10, color=:white)
    text!(ax2, alpha_th - 0.08, T_max + 0.1, text=@sprintf("T_max=%.2f", T_max),
          fontsize=10, color=:black)

    Label(fig[0, :], @sprintf("Phase Diagrams for Spherical DAM (N=%d, CPU-multithreaded)", config.N),
          fontsize=20)

    return fig, alpha_th, T_max
end

#= ============== MAIN ============== =#

function main()
    println("=" ^ 60)
    println("Phase Boundary Detection - CPU Multithreaded (Julia)")
    println("=" ^ 60)
    println()

    config = default_config()

    println("Configuration:")
    println("  N = $(config.N)")
    println("  Grid: $(config.n_alpha) × $(config.n_T) = $(config.n_alpha * config.n_T) points")
    println("  Threads: $(Threads.nthreads())")
    println("  βnet (LSE) = $(config.betanet_lse) (σ² = $(1/config.betanet_lse))")
    println("  b (LSR) = $(round(config.b, digits=4))")
    println("  MC: n_eq=$(config.n_eq), n_samp=$(config.n_samp), n_trials=$(config.n_trials)")
    println("  Precision: Float64")
    println()

    # Compute phase diagrams
    println("Computing LSE phase diagram...")
    t1 = time()
    alpha_lse, T_lse, phi_lse = compute_phase_diagram_cpu(config, :LSE)
    time_lse = time() - t1
    @printf("  LSE completed in %.1f seconds\n\n", time_lse)

    println("Computing LSR phase diagram...")
    t2 = time()
    alpha_lsr, T_lsr, phi_lsr = compute_phase_diagram_cpu(config, :LSR)
    time_lsr = time() - t2
    @printf("  LSR completed in %.1f seconds\n\n", time_lsr)

    @printf("Total CPU time: %.1f seconds\n\n", time_lse + time_lsr)

    println("Generating figures...")
    fig, alpha_th, T_max = plot_phase_diagrams(alpha_lse, T_lse, phi_lse,
                                                alpha_lsr, T_lsr, phi_lsr,
                                                config)

    @printf("LSR parameters: α_th = %.3f, T_max = %.3f\n", alpha_th, T_max)

    save("phase_boundary_cpu.png", fig, px_per_unit=2)
    save("phase_boundary_cpu.eps", fig)
    println("\nFigures saved: phase_boundary_cpu.{png,eps}")

    display(fig)

    return fig, (alpha_lse, T_lse, phi_lse), (alpha_lsr, T_lsr, phi_lsr)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

#=
GPU-Accelerated LSR Basin Stability Test (v5)
────────────────────────────────────────────────────────────────────────
v5 = v3 + deterministic-K pattern reduction

Instead of storing P full patterns and computing energy via GEMM, we:
1. Sample K ≈ M·p_tail overlaps from the truncated normal at each energy eval
2. Compute energy as H = -(N/b) * log(Σ_k [1 - b + b·φ_k]_+)
3. Keep deterministic K = round(exp(α·N)·p_tail) with N = 27.5/α

Memory: O(K·N·n_trials) instead of O(P·N·n_trials)
Time: O(K·n_chains) instead of O(P·n_chains) per energy eval

Alpha grid: 0.20:0.05:0.40 (smaller grid for testing)
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using SpecialFunctions
using Printf
using ProgressMeter

const F = Float32

# ──────────────── LSR Parameters ────────────────
const b_lsr       = F(2 + sqrt(2))
const phi_c_lsr   = F((b_lsr - 1) / b_lsr)
const PHI_MIN     = F(0.75)
const PHI_MAX     = F(1.0)
const N_TRIALS    = 256
const N_EQ        = 10000
const N_SAMP      = 5000

const alpha_vec = collect(F(0.20):F(0.05):F(0.40))
const T_vec     = F.(10 .^ range(-2, log10(2.5), length=20))
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))
const INF_ENERGY = F(1e30)

# ──────────────── Truncated Normal Sampler (Robert 1995) ────────────────
function rand_truncnorm_above(a::Float64)
    α_opt = 0.5 * (a + sqrt(a^2 + 4.0))
    while true
        z = a - log(rand()) / α_opt
        rho = exp(-0.5 * (z - α_opt)^2)
        rand() < rho && return z
    end
end

# ──────────────── Energy: Sample K overlaps from truncated normal ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              targets::CuArray{F,3}, Nf::F,
                              K::Int, z_threshold::Float64)
    # Energy computation on CPU (K is small)
    # E[i] = -(Nf/b) * log(sum of K sampled contributions)
    
    Nb = Nf / b_lsr
    tgt = Array(targets)[:, 1, :]  # [N, n_trials]
    x_cpu = Array(x)  # [N, n_T, n_trials]
    n_trials = size(x_cpu, 3)
    n_chains = n_T * n_trials
    
    E_cpu = zeros(Float64, n_chains)
    idx = 1
    
    for t in 1:n_trials
        for j in 1:n_T
            # Planted pattern contribution (exact)
            ov_target = dot(tgt[:, t], x_cpu[:, j, t]) / Float64(Nf)
            c_target = max(0.0, 1.0 - Float64(b_lsr) + Float64(b_lsr) * ov_target)
            
            # Sample K truncated-normal overlaps and sum contributions
            s = c_target  # start with target
            for k in 1:K
                z = rand_truncnorm_above(z_threshold)
                phi = z / sqrt(Float64(Nf))
                c = max(0.0, 1.0 - Float64(b_lsr) + Float64(b_lsr) * phi)
                s += c
            end
            
            # Energy
            E_cpu[idx] = s > 0 ? - Float64(Nb) * log(s) : Float64(INF_ENERGY)
            idx += 1
        end
    end
    
    # Copy to GPU
    copyto!(E, CuArray(Float32.(E_cpu)))
    return nothing
end

# ──────────────── MC Step ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  targets::CuArray{F,3}, β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::F, K::Int, z_threshold::Float64)
    CUDA.randn!(xp)
    
    # Proposal on CPU to avoid GPU broadcasts
    x_cpu = Array(x)
    xp_cpu = Array(xp)
    n_trials = size(xp_cpu, 3)
    
    for t in 1:n_trials
        for j in 1:n_T
            xp_cpu[:, j, t] .= x_cpu[:, j, t] .+ ss .* xp_cpu[:, j, t]
            nrm = norm(xp_cpu[:, j, t])
            xp_cpu[:, j, t] .*= sqrt(Float64(Nf)) / nrm
        end
    end
    copyto!(xp, xp_cpu)

    # Proposed energy
    compute_energy_lsr!(Ep, xp, targets, Nf, K, z_threshold)

    # Accept/reject on CPU
    CUDA.rand!(ra)
    ra_cpu = Array(ra)
    E_cpu = Array(E)
    Ep_cpu = Array(Ep)
    β_cpu = Array(β)
    
    acc_cpu = similar(ra_cpu, Bool)
    for i in eachindex(ra_cpu)
        acc_cpu[i] = (Ep_cpu[i] < Float32(INF_ENERGY)) & 
                     (ra_cpu[i] < exp(-(β_cpu[i] * (Ep_cpu[i] - E_cpu[i]))))
    end
    
    # Update state on CPU
    x_cpu = Array(x)
    xp_cpu = Array(xp)
    for i in eachindex(acc_cpu)
        if acc_cpu[i]
            E_cpu[i] = Ep_cpu[i]
            t = div(i - 1, n_T) + 1
            j = mod(i - 1, n_T) + 1
            x_cpu[:, j, t] .= xp_cpu[:, j, t]
        end
    end
    
    copyto!(x, x_cpu)
    copyto!(E, E_cpu)
    return nothing
end

# ──────────────── Random initialization ────────────────
function initialize_random_alignment!(x::Array{F,3}, target::Array{F,3}, N::Int)
    tgt = target[:, 1, :]
    n_trials = size(x, 3)
    
    for t in 1:n_trials
        for j in 1:n_T
            phi_init = PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F)
            x_perp = randn(F, N)
            ov = dot(tgt[:, t], x_perp) / N
            x_perp .-= ov .* tgt[:, t]
            x_perp ./= norm(x_perp)
            x[:, j, t] .= phi_init .* tgt[:, t] .+ 
                          sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
        end
    end
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSR Basin Stability Test – GPU v5 (deterministic-K reduction)")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    phi_c = Float64(phi_c_lsr)

    # CSV header
    csv_file = "basin_stab_LSR_v5.csv"
    open(csv_file, "w") do f
        write(f, "alpha")
        for T in T_vec
            write(f, @sprintf(",T%.4f", T))
        end
        write(f, "\n")
    end

    phi_grid = zeros(Float64, n_alpha, n_T)

    for (i, αf) in enumerate(alpha_vec)
        α = Float64(αf)
        
        # Compute N and K
        N = round(Int, 27.5 / α)
        Nf = F(N)
        z_threshold = phi_c * sqrt(N)
        p_tail = 0.5 * erfc(z_threshold / sqrt(2.0))
        λ = exp(α * N) * p_tail
        K = max(1, round(Int, λ))
        
        @printf("α=%.2f: N=%d, λ=%.2e, K=%d\n", α, N, λ, K)

        # Generate targets
        Random.seed!(1234 + i)
        targets_cpu = randn(F, N, N_TRIALS)
        for t in 1:N_TRIALS
            targets_cpu[:, t] .*= sqrt(Nf) / norm(targets_cpu[:, t])
        end
        targets_g = CuArray(reshape(targets_cpu, N, 1, N_TRIALS))

        # Allocate GPU arrays
        xs = CuArray(zeros(F, N, n_T, N_TRIALS))
        xps = CuArray(zeros(F, N, n_T, N_TRIALS))
        Es = CuArray(zeros(F, n_T * N_TRIALS))
        Eps = CuArray(zeros(F, n_T * N_TRIALS))
        
        beta_cpu = repeat(Float32.(1.0 ./ T_vec), N_TRIALS)
        β_gpu = CuArray(beta_cpu)
        ra = CuArray(zeros(F, n_T * N_TRIALS))

        # Initialize on CPU, copy to GPU
        xs_cpu = zeros(F, N, n_T, N_TRIALS)
        initialize_random_alignment!(xs_cpu, reshape(targets_cpu, N, 1, N_TRIALS), N)
        copyto!(xs, xs_cpu)

        # Initialize energies
        compute_energy_lsr!(Es, xs, targets_g, Nf, K, z_threshold)

        ss = adaptive_ss(N)

        # Equilibration
        println("Equilibrating...")
        prog = Progress(N_EQ, desc="Eq: ")
        for step in 1:N_EQ
            mc_step!(xs, xps, Es, Eps, targets_g, β_gpu, ra, Nf, ss, K, z_threshold)
            next!(prog)
        end
        finish!(prog)

        # Sampling
        phis_cpu = zeros(Float64, n_T * N_TRIALS)
        println("Sampling...")
        prog = Progress(N_SAMP, desc="Samp: ")
        for step in 1:N_SAMP
            mc_step!(xs, xps, Es, Eps, targets_g, β_gpu, ra, Nf, ss, K, z_threshold)
            
            # Measure φ = (target·x)/N
            xs_cpu = Array(xs)
            for t in 1:N_TRIALS
                for j in 1:n_T
                    idx = (t - 1) * n_T + j
                    phis_cpu[idx] += dot(targets_cpu[:, t], xs_cpu[:, j, t]) / Float64(Nf)
                end
            end
            next!(prog)
        end
        finish!(prog)

        phi_avg = phis_cpu ./ N_SAMP
        phi_mat = reshape(phi_avg, n_T, N_TRIALS)
        phi_grid[i, :] = vec(mean(phi_mat, dims=2))

        # Write to CSV
        open(csv_file, "a") do f
            write(f, @sprintf("%.2f", α))
            for j in 1:n_T
                write(f, @sprintf(",%.4f", phi_grid[i, j]))
            end
            write(f, "\n")
        end

        # Cleanup
        CUDA.unsafe_free!(targets_g)
        CUDA.unsafe_free!(xs)
        CUDA.unsafe_free!(xps)
        CUDA.unsafe_free!(Es)
        CUDA.unsafe_free!(Eps)
        CUDA.unsafe_free!(β_gpu)
        CUDA.unsafe_free!(ra)
    end
    
    println("\nDone. Output: $csv_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

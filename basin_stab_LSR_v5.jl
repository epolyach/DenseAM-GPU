#=
LSR Basin Stability Test – GPU v5 (deterministic-K reduced patterns)
This variant uses a deterministic number of active spurious patterns
per-α computed as K = max(1, floor(M * p_tail)) with M = exp(α*N).
N is chosen as N = round(Int, 27.5/α) (as requested).

Alpha grid: 0.20:0.05:0.40
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using SpecialFunctions   # erfc
using Printf
using ProgressMeter

const F = Float32

const b_lsr     = F(2 + sqrt(2))
const phi_c_lsr = F((b_lsr - 1) / b_lsr)
const PHI_MIN   = F(0.75)
const PHI_MAX   = F(1.0)

const alpha_vec = collect(F(0.20):F(0.05):F(0.40))
const T_vec     = F.(10 .^ range(-2, log10(2.5), length=20))
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

const N_TRIALS  = 256
const N_EQ      = 20000
const N_SAMP    = 5000

# Robert (1995) exponential-proposal method for z > a, a > 0.
function rand_truncnorm_above(a::Float64)
    α_opt = 0.5 * (a + sqrt(a^2 + 4.0))
    while true
        z = a - log(rand()) / α_opt
        rho = exp(-0.5 * (z - α_opt)^2)
        rand() < rho && return z
    end
end

const INF_ENERGY = F(1e30)

function compute_energy_lsr(x::CuArray{F,3},
                             targets::CuArray{F,3}, Nf::F,
                             K::Int, z_threshold::Float64)
    # Compute planted overlap contribution exactly
    Nb = Nf / b_lsr
    # Move targets and x to CPU and compute target overlaps
    tgt = Array(targets)[:, 1, :]
    x_cpu = Array(x)
    n_trials = size(x_cpu, 3)
    n_chains = n_T * n_trials

    # For each chain compute deterministic K truncated-normal samples and sum contributions
    phi_vals = zeros(Float64, n_chains)
    idx = 1
    for t in 1:n_trials
        for j in 1:n_T
            # planted overlap
            tgt_vec = tgt[:, t]
            ov_t = dot(tgt_vec, x_cpu[:, j, t]) / Float64(Nf)

            # sample K truncated-normal overlaps z > z_threshold (scale sqrt(N))
            sum_contrib = 0.0
            for k in 1:K
                z = rand_truncnorm_above(z_threshold)
                φ = z / sqrt(Float64(Nf))
                # Use same contribution formula: max(0, 1 - b + b*φ)
                c = max(0.0, 1.0 - Float64(b_lsr) + Float64(b_lsr) * φ)
                sum_contrib += c
            end

            # include planted pattern contribution
            planted = max(0.0, 1.0 - Float64(b_lsr) + Float64(b_lsr) * ov_t)
            s = planted + sum_contrib
            phi_vals[idx] = s > 0 ? - (Float64(Nf) / Float64(b_lsr)) * log(s) : Float64(INF_ENERGY)
            idx += 1
        end
    end

    # return as CuVector
    return CuVector{F}(Float32.(phi_vals))
end

function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  targets::CuArray{F,3}, β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::F, K::Int, z_threshold::Float64)
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    Ep = compute_energy_lsr(xp, targets, Nf, K, z_threshold)

    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-(β * (Ep - E))))
    n_trials = length(β) ÷ n_T
    a3 = reshape(acc, 1, n_T, n_trials)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

function main()
    !CUDA.functional() && error("CUDA not available")

    println("LSR Basin Stability Test – GPU v5 (deterministic-K)")
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
        # N rule: 27.5 / α
        N = max(2, round(Int, 27.5 / α))
        Nf = F(N)
        # threshold in z (standard normal scale sqrt(N/2) used earlier); here use z = phi_c * sqrt(N)
        z_threshold = phi_c * sqrt(N)

        # tail prob and M
        p_tail = 0.5 * erfc(z_threshold / sqrt(2.0))
        logM = α * N
        λ = exp(logM) * p_tail
        K = max(1, floor(Int, λ))

        @printf("α=%.2f: N=%d, p_tail=%.3e, K=%d\n", α, N, p_tail, K)

        # Generate targets per trial (store exactly)
        Random.seed!(1234 + i)
        targets_cpu = randn(F, N, N_TRIALS)
        for t in 1:N_TRIALS
            targets_cpu[:, t] .*= sqrt(Nf) / norm(targets_cpu[:, t])
        end
        targets_g = CuArray(reshape(targets_cpu, N, 1, N_TRIALS))

        # states
        xs = CUDA.zeros(F, N, n_T, N_TRIALS)
        xps = CUDA.zeros(F, N, n_T, N_TRIALS)
        Es = CUDA.zeros(F, n_T * N_TRIALS)
        Eps = CUDA.zeros(F, n_T * N_TRIALS)  # will be reassigned by compute_energy_lsr
        beta_cpu = repeat(Float32.(1.0 ./ T_vec), N_TRIALS)
        β_gpu = CuVector{F}(beta_cpu)
        ra = CUDA.zeros(F, n_T * N_TRIALS)

        # initialize states near target
        for t in 1:N_TRIALS
            for j in 1:n_T
                xs[:, j, t] .= targets_cpu[:, t] .+ F(0.05) .* randn(F, N)
                nrm = norm(xs[:, j, t])
                xs[:, j, t] .*= sqrt(Nf) / nrm
            end
        end
        Es = compute_energy_lsr(xs, targets_g, Nf, K, z_threshold)

        # equilibration
        ss = max(F(0.1), F(2.4) / sqrt(F(N)))
        prog = Progress(N_EQ, desc="Equilibration: ")
        for step in 1:N_EQ
            mc_step!(xs, xps, Es, Eps, targets_g, β_gpu, ra, Nf, ss, K, z_threshold)
            next!(prog)
        end
        finish!(prog)

        # sampling
        phis_cpu = zeros(Float64, n_T * N_TRIALS)
        prog = Progress(N_SAMP, desc="Sampling: ")
        for step in 1:N_SAMP
            mc_step!(xs, xps, Es, Eps, targets_g, β_gpu, ra, Nf, ss, K, z_threshold)
            # measure φ = (target·x)/N
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

        # append to CSV
        open(csv_file, "a") do f
            write(f, @sprintf("%.2f", α))
            for j in 1:n_T
                write(f, @sprintf(",%.4f", phi_grid[i, j]))
            end
            write(f, "\n")
        end

        CUDA.unsafe_free!(targets_g)
        CUDA.unsafe_free!(xs)
        CUDA.unsafe_free!(xps)
        CUDA.unsafe_free!(Es)
        if Eps isa CuVector
            CUDA.unsafe_free!(Eps)
        end
        CUDA.unsafe_free!(β_gpu)
        CUDA.unsafe_free!(ra)
    end
    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

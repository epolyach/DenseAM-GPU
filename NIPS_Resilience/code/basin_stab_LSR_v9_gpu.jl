#=
GPU-Accelerated LSR Diagnostic Test (v9) — METASTABILITY PROBE
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v9.jl

v9: Targeted diagnostic at selected (α,T) points in the anomalous region
  where q > φ² (possible metastability or mixture states).

  Three diagnostics from one long MC run:
  1. Equilibration convergence: φ measured at checkpoints (16k, 64k, 256k steps)
  2. Per-trial φ distribution: individual trial values (not averaged), for histograms
  3. φ(t) trajectory: φ recorded every TRAJ_STRIDE steps for representative trials

  Also records: q_12, φ_max_other per trial.

  Two replicas per disorder sample as in v8.

Probe points:
  Anomalous region: (0.27, 0.40), (0.28, 0.50), (0.30, 0.30)
  Control R:        (0.22, 0.50)
  Control P:        (0.32, 1.00)

Output:
  v9_summary.csv          — per-point averages at each checkpoint
  v9_per_trial.csv        — per-trial φ, q, φ_max_other (for histograms)
  v9_trajectory_*.csv     — φ(t) for first N_TRAJ trials at each point
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using CUDA: @allowscalar
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ──────────────── Precision Settings ────────────────
const USE_FLOAT16 = false
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Parameters ────────────────
const b_lsr       = F(2 + sqrt(2))
const PHI_MIN     = F(0.75)
const PHI_MAX     = F(1.0)
const INF_ENERGY  = F(1e30)

# Long run: 256k total steps, checkpoints at 16k, 64k, 256k
const N_TOTAL     = 2^18              # 262144 total MC steps
const CHECKPOINTS = [2^14, 2^16, 2^18]  # 16384, 65536, 262144
const N_SAMP      = 2^12              # 4096 sampling steps after each checkpoint
const N_TRIALS    = 128               # trials per probe point (64 disorder samples × 2 replicas)
const N_DISORDER  = N_TRIALS ÷ 2
const TRAJ_STRIDE = 16                # record φ every this many steps during sampling
const N_TRAJ      = 8                 # number of trials for which to save full trajectory

# Probe points: (α, T)
const PROBE_POINTS = [
    (F(0.27), F(0.40)),   # anomalous
    (F(0.28), F(0.50)),   # anomalous
    (F(0.30), F(0.30)),   # anomalous
    (F(0.22), F(0.50)),   # control R
    (F(0.32), F(1.00)),   # control P
]

const MIN_PAT = 20000
const MAX_PAT = 500000
const ind     = 10

# ──────────────── LSR Energy ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,2},
                              patterns::CuMatrix{F}, overlap::CuVector{F},
                              Nf::F)
    Nb = Nf / b_lsr
    mul!(overlap, patterns', vec(x))
    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap)
    s_cpu = Float32(s)
    @allowscalar E[1] = s_cpu > zero(F) ? -Nb * log(s_cpu) : INF_ENERGY
    return nothing
end

# ──────────────── MC Step (single chain) ────────────────
function mc_step_single!(x::CuMatrix{F}, xp::CuMatrix{F},
                          E::CuVector{F}, Ep::CuVector{F},
                          pat::CuMatrix{F}, ov::CuVector{F},
                          β::F, Nf::F, σ::F)
    N = size(x, 1)
    CUDA.randn!(xp)
    @. xp = x + σ * xp
    nrm = sqrt(sum(xp .^ 2))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    r = rand(F)
    @allowscalar begin
        if Ep[1] < INF_ENERGY && r < exp(-β * (Ep[1] - E[1]))
            x .= xp
            E .= Ep
        end
    end
    return nothing
end

# ──────────────── Initialize with controlled φ ────────────────
function initialize_near_pattern(target::Vector{F}, N::Int, phi_init::F)
    x_perp = randn(F, N)
    ov = dot(target, x_perp) / N
    x_perp .-= ov .* target
    x_perp ./= norm(x_perp)
    x = phi_init .* target .+ sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
    return x
end

# ──────────────── Compute φ_max_other on CPU ────────────────
function compute_phi_max_other_cpu(x::Vector{F}, patterns::Matrix{F}, Nf::F)
    overlaps = patterns' * x ./ Nf
    overlaps[1] = F(-Inf)  # exclude target
    return maximum(overlaps)
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)
    println("=" ^ 70)
    println("LSR Diagnostic Test – v9 (metastability probe)")
    println("  Probe points: $(length(PROBE_POINTS))")
    println("  Total MC steps: $N_TOTAL")
    println("  Checkpoints: $CHECKPOINTS")
    println("  Sampling after checkpoint: $N_SAMP steps")
    println("  Trials: $N_TRIALS ($N_DISORDER disorder samples × 2 replicas)")
    println("  Trajectory stride: $TRAJ_STRIDE")
    println("=" ^ 70)

    # Open output files
    f_summary = open("v9_summary.csv", "w")
    write(f_summary, "alpha,T,checkpoint,phi_mean,phi_std,q_mean,q_std,phimax_mean,phimax_std\n")

    f_trials = open("v9_per_trial.csv", "w")
    write(f_trials, "alpha,T,checkpoint,trial,phi,q12,phi_max_other\n")

    for (pi, (α, T)) in enumerate(PROBE_POINTS)
        # Compute N and P for this α
        n_pat_interp = (MIN_PAT^(1/ind) + (MAX_PAT^(1/ind) - MIN_PAT^(1/ind)) *
                        (Float64(α) - 0.20) / (0.35 - 0.20))^ind
        P = round(Int, n_pat_interp)
        N = max(round(Int, log(P) / Float64(α)), 2)
        Nf = F(N)
        β = F(1 / T)
        σ = F(2.4 * T / sqrt(Float64(N)))

        @printf("\n── Point %d/%d: α=%.3f, T=%.3f (N=%d, P=%d) ──\n",
                pi, length(PROBE_POINTS), α, T, N, P)

        # Open trajectory file for this point
        f_traj = open(@sprintf("v9_trajectory_a%.3f_T%.3f.csv", α, T), "w")
        write(f_traj, "trial,step,phi_a,phi_b,q12\n")

        # Per-trial results at each checkpoint: [checkpoint][trial]
        n_cp = length(CHECKPOINTS)
        trial_phi    = zeros(Float64, n_cp, N_TRIALS)
        trial_q      = zeros(Float64, n_cp, N_TRIALS)
        trial_phimax = zeros(Float64, n_cp, N_TRIALS)

        # Run each disorder sample sequentially (small N, fits on GPU easily)
        for d in 1:N_DISORDER
            Random.seed!(42000 + pi * 1000 + d)

            # Generate patterns
            p_cpu = randn(F, N, P)
            for j in 1:P
                c = @view p_cpu[:, j]
                c .*= sqrt(Nf) / norm(c)
            end
            target_cpu = p_cpu[:, 1]

            pat_g = CuMatrix{F}(p_cpu)
            ov_g  = CUDA.zeros(F, P)

            # Initialize two replicas
            Random.seed!(100000 + pi * 1000 + d)
            phi_init_a = PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F)
            xa_cpu = initialize_near_pattern(target_cpu, N, phi_init_a)

            Random.seed!(200000 + pi * 1000 + d)
            phi_init_b = PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F)
            xb_cpu = initialize_near_pattern(target_cpu, N, phi_init_b)

            xa_g  = CuMatrix{F}(reshape(xa_cpu, N, 1))
            xb_g  = CuMatrix{F}(reshape(xb_cpu, N, 1))
            xp_g  = CUDA.zeros(F, N, 1)
            Ea_g  = CUDA.zeros(F, 1)
            Eb_g  = CUDA.zeros(F, 1)
            Ep_g  = CUDA.zeros(F, 1)

            # Compute initial energies
            compute_energy_lsr!(Ea_g, xa_g, pat_g, ov_g, Nf)
            compute_energy_lsr!(Eb_g, xb_g, pat_g, ov_g, Nf)

            # Run MC steps with checkpoints
            checkpoint_idx = 1
            step_count = 0

            for cp in CHECKPOINTS
                # Equilibration: run from current position to checkpoint
                n_eq_steps = cp - step_count
                for s in 1:n_eq_steps
                    mc_step_single!(xa_g, xp_g, Ea_g, Ep_g, pat_g, ov_g, β, Nf, σ)
                    mc_step_single!(xb_g, xp_g, Eb_g, Ep_g, pat_g, ov_g, β, Nf, σ)
                end
                step_count = cp

                # Sampling phase: N_SAMP steps
                phi_acc_a = 0.0
                phi_acc_b = 0.0
                q_acc     = 0.0
                phimax_acc = 0.0

                tgt_g = CuMatrix{F}(reshape(target_cpu, N, 1))

                for s in 1:N_SAMP
                    mc_step_single!(xa_g, xp_g, Ea_g, Ep_g, pat_g, ov_g, β, Nf, σ)
                    mc_step_single!(xb_g, xp_g, Eb_g, Ep_g, pat_g, ov_g, β, Nf, σ)

                    # Measure
                    φa = Float64(sum(Array(tgt_g .* xa_g)) / Nf)
                    φb = Float64(sum(Array(tgt_g .* xb_g)) / Nf)
                    q12 = Float64(sum(Array(xa_g .* xb_g)) / Nf)

                    phi_acc_a += φa
                    phi_acc_b += φb
                    q_acc     += q12

                    # φ_max_other (CPU, from replica a)
                    xa_cpu_now = vec(Array(xa_g))
                    pm = Float64(compute_phi_max_other_cpu(xa_cpu_now, p_cpu, Nf))
                    phimax_acc += pm

                    # Trajectory recording for first N_TRAJ disorder samples
                    if d <= N_TRAJ ÷ 2 && s % TRAJ_STRIDE == 0
                        trial_a = 2*(d-1) + 1
                        trial_b = 2*(d-1) + 2
                        write(f_traj, @sprintf("%d,%d,%.6f,%.6f,%.6f\n",
                              trial_a, step_count + s, φa, φb, q12))
                    end
                end

                # Store per-trial averages
                trial_a = 2*(d-1) + 1
                trial_b = 2*(d-1) + 2
                φ_avg = (phi_acc_a + phi_acc_b) / (2 * N_SAMP)
                q_avg = q_acc / N_SAMP
                pm_avg = phimax_acc / N_SAMP

                ci = findfirst(==(cp), CHECKPOINTS)
                trial_phi[ci, trial_a]    = phi_acc_a / N_SAMP
                trial_phi[ci, trial_b]    = phi_acc_b / N_SAMP
                trial_q[ci, trial_a]      = q_avg
                trial_q[ci, trial_b]      = q_avg
                trial_phimax[ci, trial_a] = pm_avg
                trial_phimax[ci, trial_b] = pm_avg

                # Write per-trial data
                write(f_trials, @sprintf("%.3f,%.3f,%d,%d,%.6f,%.6f,%.6f\n",
                      α, T, cp, trial_a, trial_phi[trial_a], q_avg, pm_avg))
                write(f_trials, @sprintf("%.3f,%.3f,%d,%d,%.6f,%.6f,%.6f\n",
                      α, T, cp, trial_b, trial_phi[trial_b], q_avg, pm_avg))

                # Don't reset step_count — continue from current state
                # The next checkpoint equilibration continues from here
            end

            # Free GPU memory for this disorder sample
            pat_g = nothing; ov_g = nothing
            xa_g = nothing; xb_g = nothing; xp_g = nothing
            Ea_g = nothing; Eb_g = nothing; Ep_g = nothing
        end

        close(f_traj)

        # Write summary for each checkpoint
        for (ci, cp) in enumerate(CHECKPOINTS)
            # Gather per-trial data for this checkpoint from the CSV
            # (already written above; compute summary here)
            # Re-read is wasteful; compute on the fly instead
        end

        # Write summary for each checkpoint
        for (ci, cp) in enumerate(CHECKPOINTS)
            φ_mean  = mean(trial_phi[ci, :])
            φ_std   = std(trial_phi[ci, :])
            q_mean  = mean(trial_q[ci, :])
            q_std   = std(trial_q[ci, :])
            pm_mean = mean(trial_phimax[ci, :])
            pm_std  = std(trial_phimax[ci, :])

            write(f_summary, @sprintf("%.3f,%.3f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                  α, T, cp, φ_mean, φ_std, q_mean, q_std, pm_mean, pm_std))

            @printf("  checkpoint %6d: φ=%.4f±%.4f, q=%.4f±%.4f, φ_max=%.4f±%.4f\n",
                    cp, φ_mean, φ_std, q_mean, q_std, pm_mean, pm_std)
        end
    end

    close(f_summary)
    close(f_trials)

    println("\n" * "=" ^ 70)
    println("Output files:")
    println("  v9_summary.csv          — per-point averages")
    println("  v9_per_trial.csv        — per-trial values (for histograms)")
    println("  v9_trajectory_*.csv     — φ(t) trajectories")
    println("=" ^ 70)
end

main()

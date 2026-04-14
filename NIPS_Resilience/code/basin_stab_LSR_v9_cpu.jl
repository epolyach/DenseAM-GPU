#=
LSR Diagnostic Test (v9) — METASTABILITY PROBE (CPU version)
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v9.jl

v9: Targeted diagnostic at selected (α,T) points in the anomalous region
  where q > φ². Runs on CPU (N is small, GPU overhead dominates).

  Three diagnostics from one long MC run:
  1. Equilibration convergence: φ measured at checkpoints (16k, 64k, 256k steps)
  2. Per-trial φ distribution: individual trial values (not averaged), for histograms
  3. φ(t) trajectory: φ recorded every TRAJ_STRIDE steps for representative trials

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

using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ──────────────── Precision Settings ────────────────
const F = Float32

# ──────────────── Parameters ────────────────
const b_lsr       = F(2 + sqrt(2))
const PHI_MIN     = F(0.75)
const PHI_MAX     = F(1.0)
const INF_ENERGY  = F(1e30)

const CHECKPOINTS = [2^14, 2^16, 2^18]  # 16384, 65536, 262144
const N_SAMP      = 2^12              # 4096 sampling steps after each checkpoint
const N_DISORDER  = 64
const N_TRIALS    = N_DISORDER * 2    # 128
const TRAJ_STRIDE = 16
const N_TRAJ_DIS  = 4                 # disorder samples for which to save trajectories

# Probe points: (α, T)
const PROBE_POINTS = [
    (F(0.27), F(0.40)),
    (F(0.28), F(0.50)),
    (F(0.30), F(0.30)),
    (F(0.22), F(0.50)),
    (F(0.32), F(1.00)),
]

const MIN_PAT = 20000
const MAX_PAT = 500000
const ind     = 10

# ──────────────── LSR Energy (CPU) ────────────────
function compute_energy_cpu(x::Vector{F}, patterns::Matrix{F}, overlap::Vector{F}, Nf::F)
    Nb = Nf / b_lsr
    mul!(overlap, patterns', x)
    @inbounds for i in eachindex(overlap)
        v = one(F) - b_lsr + b_lsr * overlap[i] / Nf
        overlap[i] = v > zero(F) ? v : zero(F)
    end
    s = sum(overlap)
    return s > zero(F) ? -Nb * log(s) : INF_ENERGY
end

# ──────────────── MC Step (CPU) ────────────────
function mc_step_cpu!(x::Vector{F}, xp::Vector{F}, E::Ref{F},
                      pat::Matrix{F}, ov::Vector{F},
                      β::F, Nf::F, σ::F)
    N = length(x)
    # Propose
    randn!(xp)
    @inbounds for i in 1:N
        xp[i] = x[i] + σ * xp[i]
    end
    nrm = norm(xp)
    scale = sqrt(Nf) / nrm
    @inbounds for i in 1:N
        xp[i] *= scale
    end

    Ep = compute_energy_cpu(xp, pat, ov, Nf)

    # Accept/reject
    if Ep < INF_ENERGY && rand(F) < exp(-β * (Ep - E[]))
        x .= xp
        E[] = Ep
    end
    return nothing
end

# ──────────────── Initialize ────────────────
function initialize_near_pattern(target::Vector{F}, N::Int, phi_init::F)
    x_perp = randn(F, N)
    ov = dot(target, x_perp) / N
    x_perp .-= ov .* target
    x_perp ./= norm(x_perp)
    return phi_init .* target .+ sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
end

# ──────────────── φ_max_other (CPU) ────────────────
function phi_max_other_cpu(x::Vector{F}, patterns::Matrix{F}, ov::Vector{F}, Nf::F)
    mul!(ov, patterns', x)
    @inbounds ov[1] = F(-Inf)  # exclude target
    return maximum(ov) / Nf
end

# ──────────────── Main ────────────────
function main()
    println("=" ^ 70)
    println("LSR Diagnostic Test – v9 (CPU, metastability probe)")
    println("  Probe points: $(length(PROBE_POINTS))")
    println("  Checkpoints: $CHECKPOINTS")
    println("  Sampling after checkpoint: $N_SAMP steps")
    println("  Disorder samples: $N_DISORDER (× 2 replicas = $N_TRIALS trials)")
    println("=" ^ 70)

    f_summary = open("v9_summary.csv", "w")
    write(f_summary, "alpha,T,checkpoint,phi_mean,phi_std,q_mean,q_std,phimax_mean,phimax_std,q_minus_phi2_mean\n")

    f_trials = open("v9_per_trial.csv", "w")
    write(f_trials, "alpha,T,checkpoint,trial,phi_a,phi_b,q12,phi_max_other_a\n")

    for (pi, (α, T)) in enumerate(PROBE_POINTS)
        # Compute N and P
        frac = clamp((Float64(α) - 0.20) / (0.35 - 0.20), 0.0, 1.0)
        n_pat = (MIN_PAT^(1/ind) + (MAX_PAT^(1/ind) - MIN_PAT^(1/ind)) * frac)^ind
        P = round(Int, n_pat)
        N = max(round(Int, log(P) / Float64(α)), 2)
        Nf = F(N)
        β = F(1 / T)
        σ = F(2.4 * T / sqrt(Float64(N)))

        @printf("\n── Point %d/%d: α=%.3f, T=%.3f (N=%d, P=%d) ──\n",
                pi, length(PROBE_POINTS), α, T, N, P)

        # Trajectory file
        f_traj = open(@sprintf("v9_trajectory_a%.3f_T%.3f.csv", α, T), "w")
        write(f_traj, "disorder,step,phi_a,phi_b,q12\n")

        # Per-checkpoint storage
        n_cp = length(CHECKPOINTS)
        cp_phi_a   = zeros(Float64, n_cp, N_DISORDER)
        cp_phi_b   = zeros(Float64, n_cp, N_DISORDER)
        cp_q       = zeros(Float64, n_cp, N_DISORDER)
        cp_phimax  = zeros(Float64, n_cp, N_DISORDER)

        prog = Progress(N_DISORDER, desc="  Disorder: ")

        for d in 1:N_DISORDER
            Random.seed!(42000 + pi * 1000 + d)

            # Generate patterns on CPU
            p_cpu = randn(F, N, P)
            for j in 1:P
                c = @view p_cpu[:, j]
                c .*= sqrt(Nf) / norm(c)
            end
            target = p_cpu[:, 1]

            # Workspace
            ov = zeros(F, P)
            xp_a = zeros(F, N)
            xp_b = zeros(F, N)

            # Initialize replicas
            Random.seed!(100000 + pi * 1000 + d)
            xa = initialize_near_pattern(target, N, PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F))
            Random.seed!(200000 + pi * 1000 + d)
            xb = initialize_near_pattern(target, N, PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F))

            Ea = Ref(compute_energy_cpu(xa, p_cpu, ov, Nf))
            Eb = Ref(compute_energy_cpu(xb, p_cpu, ov, Nf))

            # Run through checkpoints
            step_done = 0
            for (ci, cp) in enumerate(CHECKPOINTS)
                # Equilibration: run to checkpoint
                n_eq = cp - step_done
                for s in 1:n_eq
                    mc_step_cpu!(xa, xp_a, Ea, p_cpu, ov, β, Nf, σ)
                    mc_step_cpu!(xb, xp_b, Eb, p_cpu, ov, β, Nf, σ)
                end
                step_done = cp

                # Sampling
                phi_acc_a = 0.0
                phi_acc_b = 0.0
                q_acc     = 0.0
                pm_acc    = 0.0

                for s in 1:N_SAMP
                    mc_step_cpu!(xa, xp_a, Ea, p_cpu, ov, β, Nf, σ)
                    mc_step_cpu!(xb, xp_b, Eb, p_cpu, ov, β, Nf, σ)

                    φa = Float64(dot(target, xa) / Nf)
                    φb = Float64(dot(target, xb) / Nf)
                    q12 = Float64(dot(xa, xb) / Nf)
                    pm = Float64(phi_max_other_cpu(xa, p_cpu, ov, Nf))

                    phi_acc_a += φa
                    phi_acc_b += φb
                    q_acc     += q12
                    pm_acc    += pm

                    # Trajectory for first N_TRAJ_DIS disorder samples
                    if d <= N_TRAJ_DIS && s % TRAJ_STRIDE == 0
                        write(f_traj, @sprintf("%d,%d,%.6f,%.6f,%.6f\n",
                              d, step_done + s, φa, φb, q12))
                    end
                end
                step_done += N_SAMP

                # Store per-disorder averages
                cp_phi_a[ci, d]  = phi_acc_a / N_SAMP
                cp_phi_b[ci, d]  = phi_acc_b / N_SAMP
                cp_q[ci, d]      = q_acc / N_SAMP
                cp_phimax[ci, d] = pm_acc / N_SAMP

                # Write per-trial
                write(f_trials, @sprintf("%.3f,%.3f,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                      α, T, cp, 2*d-1, cp_phi_a[ci,d], cp_phi_b[ci,d], cp_q[ci,d], cp_phimax[ci,d]))
            end

            next!(prog)
        end
        finish!(prog)
        close(f_traj)

        # Summary
        for (ci, cp) in enumerate(CHECKPOINTS)
            φ_all = vcat(cp_phi_a[ci,:], cp_phi_b[ci,:])
            q_all = cp_q[ci,:]
            pm_all = cp_phimax[ci,:]
            φ_mean = mean(φ_all)
            q_mean = mean(q_all)
            qmφ2 = mean(q_all .- ((cp_phi_a[ci,:] .+ cp_phi_b[ci,:])./2).^2)

            write(f_summary, @sprintf("%.3f,%.3f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                  α, T, cp, φ_mean, std(φ_all), q_mean, std(q_all),
                  mean(pm_all), std(pm_all), qmφ2))

            @printf("  cp=%6d: φ=%.4f±%.4f  q=%.4f±%.4f  q-φ²=%.4f  φ_max=%.4f\n",
                    cp, φ_mean, std(φ_all), q_mean, std(q_all), qmφ2, mean(pm_all))
        end
    end

    close(f_summary)
    close(f_trials)

    println("\n" * "=" ^ 70)
    println("Output:")
    println("  v9_summary.csv")
    println("  v9_per_trial.csv")
    println("  v9_trajectory_*.csv")
    println("=" ^ 70)
end

main()

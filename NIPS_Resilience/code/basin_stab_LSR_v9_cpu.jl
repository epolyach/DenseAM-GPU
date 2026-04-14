#=
LSR Diagnostic Test (v9) — METASTABILITY PROBE (multithreaded CPU)
────────────────────────────────────────────────────────────────────────
Usage:
  julia -t auto basin_stab_LSR_v9_cpu.jl    # use all cores
  julia -t 16   basin_stab_LSR_v9_cpu.jl    # use 16 threads

v9: Targeted diagnostic at selected (α,T) points in the anomalous region
  where q > φ². Runs on CPU with threading over disorder samples.

  Three diagnostics from one long MC run:
  1. Equilibration convergence: φ at checkpoints (16k, 64k, 256k steps)
  2. Per-trial φ distribution: individual trial values for histograms
  3. φ(t) trajectory: φ recorded every TRAJ_STRIDE steps

Output:
  v9_summary.csv          — per-point averages at each checkpoint
  v9_per_trial.csv        — per-trial φ, q, φ_max_other
  v9_trajectory_*.csv     — φ(t) for first N_TRAJ_DIS disorder samples
────────────────────────────────────────────────────────────────────────
=#

using Random
using Statistics
using LinearAlgebra
using Printf

# ──────────────── Parameters ────────────────
const F = Float32
const b_lsr       = F(2 + sqrt(2))
const PHI_MIN     = F(0.75)
const PHI_MAX     = F(1.0)
const INF_ENERGY  = F(1e30)

const CHECKPOINTS = [2^14, 2^16, 2^18]  # 16384, 65536, 262144
const N_SAMP      = 2^12              # 4096 sampling steps after each checkpoint
const N_DISORDER  = 64
const TRAJ_STRIDE = 16
const N_TRAJ_DIS  = 4                 # disorder samples for trajectory output

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

# ──────────────── LSR Energy (CPU, inlined) ────────────────
@inline function compute_energy(x::Vector{F}, pat_t::Matrix{F}, ov::Vector{F}, Nf::F)
    # pat_t is P×N (transposed patterns for fast mat-vec)
    mul!(ov, pat_t, x)
    s = zero(F)
    @inbounds @simd for i in eachindex(ov)
        v = one(F) - b_lsr + b_lsr * ov[i] / Nf
        s += max(zero(F), v)
    end
    Nb = Nf / b_lsr
    return s > zero(F) ? -Nb * log(s) : INF_ENERGY
end

# ──────────────── MC Step (CPU, no allocations) ────────────────
@inline function mc_step!(x::Vector{F}, xp::Vector{F}, E::Ref{F},
                           pat_t::Matrix{F}, ov::Vector{F},
                           β::F, Nf::F, σ::F, rng::AbstractRNG)
    N = length(x)
    # Propose: xp = x + σ * randn, then project to sphere
    nrm_sq = zero(F)
    @inbounds for i in 1:N
        xp[i] = x[i] + σ * randn(rng, F)
        nrm_sq += xp[i] * xp[i]
    end
    scale = sqrt(Nf / nrm_sq)
    @inbounds @simd for i in 1:N
        xp[i] *= scale
    end

    Ep = compute_energy(xp, pat_t, ov, Nf)

    if Ep < INF_ENERGY && rand(rng, F) < exp(-β * (Ep - E[]))
        @inbounds @simd for i in 1:N
            x[i] = xp[i]
        end
        E[] = Ep
    end
    return nothing
end

# ──────────────── Initialize ────────────────
function init_near_pattern(target::Vector{F}, N::Int, phi_init::F, rng::AbstractRNG)
    x_perp = randn(rng, F, N)
    ov = dot(target, x_perp) / N
    @inbounds for i in 1:N
        x_perp[i] -= ov * target[i]
    end
    x_perp ./= norm(x_perp)
    x = similar(target)
    sq = sqrt(1 - phi_init^2) * sqrt(F(N))
    @inbounds for i in 1:N
        x[i] = phi_init * target[i] + sq * x_perp[i]
    end
    return x
end

# ──────────────── φ_max_other ────────────────
@inline function phi_max_other(x::Vector{F}, pat_t::Matrix{F}, ov::Vector{F}, Nf::F)
    mul!(ov, pat_t, x)
    mx = F(-Inf)
    @inbounds for i in 2:length(ov)  # skip pattern 1 (target)
        v = ov[i] / Nf
        mx = max(mx, v)
    end
    return mx
end

# ──────────────── Run one disorder sample (all checkpoints) ────────────────
struct TrialResult
    phi_a::Vector{Float64}   # [n_cp]
    phi_b::Vector{Float64}
    q::Vector{Float64}
    phimax::Vector{Float64}
    traj::Vector{Tuple{Int,Float64,Float64,Float64}}  # (step, φa, φb, q12)
end

function run_disorder(pi::Int, d::Int, α::F, T::F, N::Int, P::Int,
                       save_traj::Bool)
    Nf = F(N)
    β = F(1 / T)
    σ = F(2.4 * T / sqrt(Float64(N)))
    n_cp = length(CHECKPOINTS)

    # RNG for this disorder sample
    rng_pat = MersenneTwister(42000 + pi * 1000 + d)

    # Generate patterns and pre-transpose for fast mat-vec
    p_cpu = randn(rng_pat, F, N, P)
    @inbounds for j in 1:P
        c = @view p_cpu[:, j]
        c .*= sqrt(Nf) / norm(c)
    end
    pat_t = collect(p_cpu')  # P×N, row-major access for mat-vec
    target = p_cpu[:, 1]

    # Workspace (thread-local, no allocation in hot loop)
    ov   = zeros(F, P)
    xp_a = zeros(F, N)
    xp_b = zeros(F, N)

    # Initialize replicas with separate RNGs
    rng_a = MersenneTwister(100000 + pi * 1000 + d)
    rng_b = MersenneTwister(200000 + pi * 1000 + d)
    xa = init_near_pattern(target, N, PHI_MIN + (PHI_MAX - PHI_MIN) * rand(rng_a, F), rng_a)
    xb = init_near_pattern(target, N, PHI_MIN + (PHI_MAX - PHI_MIN) * rand(rng_b, F), rng_b)

    Ea = Ref(compute_energy(xa, pat_t, ov, Nf))
    Eb = Ref(compute_energy(xb, pat_t, ov, Nf))

    # Results
    res_phi_a  = zeros(Float64, n_cp)
    res_phi_b  = zeros(Float64, n_cp)
    res_q      = zeros(Float64, n_cp)
    res_phimax = zeros(Float64, n_cp)
    traj_data  = Tuple{Int,Float64,Float64,Float64}[]

    step_done = 0
    for (ci, cp) in enumerate(CHECKPOINTS)
        # Equilibration
        n_eq = cp - step_done
        for s in 1:n_eq
            mc_step!(xa, xp_a, Ea, pat_t, ov, β, Nf, σ, rng_a)
            mc_step!(xb, xp_b, Eb, pat_t, ov, β, Nf, σ, rng_b)
        end
        step_done = cp

        # Sampling
        phi_acc_a = 0.0
        phi_acc_b = 0.0
        q_acc     = 0.0
        pm_acc    = 0.0

        for s in 1:N_SAMP
            mc_step!(xa, xp_a, Ea, pat_t, ov, β, Nf, σ, rng_a)
            mc_step!(xb, xp_b, Eb, pat_t, ov, β, Nf, σ, rng_b)

            φa  = Float64(dot(target, xa) / Nf)
            φb  = Float64(dot(target, xb) / Nf)
            q12 = Float64(dot(xa, xb) / Nf)

            phi_acc_a += φa
            phi_acc_b += φb
            q_acc     += q12

            # φ_max_other only during last checkpoint sampling (expensive)
            if ci == n_cp
                pm_acc += Float64(phi_max_other(xa, pat_t, ov, Nf))
            end

            # Trajectory
            if save_traj && s % TRAJ_STRIDE == 0
                push!(traj_data, (step_done + s, φa, φb, q12))
            end
        end
        step_done += N_SAMP

        res_phi_a[ci]  = phi_acc_a / N_SAMP
        res_phi_b[ci]  = phi_acc_b / N_SAMP
        res_q[ci]      = q_acc / N_SAMP
        res_phimax[ci] = ci == n_cp ? pm_acc / N_SAMP : NaN
    end

    return TrialResult(res_phi_a, res_phi_b, res_q, res_phimax, traj_data)
end

# ──────────────── Main ────────────────
function main()
    nt = Threads.nthreads()
    println("=" ^ 70)
    println("LSR Diagnostic Test – v9 (CPU, $nt threads)")
    println("  Probe points: $(length(PROBE_POINTS))")
    println("  Checkpoints: $CHECKPOINTS")
    println("  Sampling: $N_SAMP steps per checkpoint")
    println("  Disorder: $N_DISORDER samples (× 2 replicas)")
    println("=" ^ 70)

    f_summary = open("v9_summary.csv", "w")
    write(f_summary, "alpha,T,checkpoint,phi_mean,phi_std,q_mean,q_std,phimax_mean,phimax_std,q_minus_phi2_mean\n")

    f_trials = open("v9_per_trial.csv", "w")
    write(f_trials, "alpha,T,checkpoint,disorder,phi_a,phi_b,q12,phi_max_other\n")

    for (pi, (α, T)) in enumerate(PROBE_POINTS)
        frac = clamp((Float64(α) - 0.20) / (0.35 - 0.20), 0.0, 1.0)
        n_pat = (MIN_PAT^(1/ind) + (MAX_PAT^(1/ind) - MIN_PAT^(1/ind)) * frac)^ind
        P = round(Int, n_pat)
        N = max(round(Int, log(P) / Float64(α)), 2)

        @printf("\n── Point %d/%d: α=%.3f, T=%.3f (N=%d, P=%d) ──\n",
                pi, length(PROBE_POINTS), α, T, N, P)

        # Run all disorder samples in parallel
        results = Vector{TrialResult}(undef, N_DISORDER)
        t0 = time()

        Threads.@threads for d in 1:N_DISORDER
            results[d] = run_disorder(pi, d, α, T, N, P, d <= N_TRAJ_DIS)
        end

        elapsed = time() - t0
        @printf("  Elapsed: %.1f s (%.1f ms/disorder)\n", elapsed, 1000*elapsed/N_DISORDER)

        # Write trajectory files (sequential, from first N_TRAJ_DIS)
        f_traj = open(@sprintf("v9_trajectory_a%.3f_T%.3f.csv", α, T), "w")
        write(f_traj, "disorder,step,phi_a,phi_b,q12\n")
        for d in 1:min(N_TRAJ_DIS, N_DISORDER)
            for (step, φa, φb, q12) in results[d].traj
                write(f_traj, @sprintf("%d,%d,%.6f,%.6f,%.6f\n", d, step, φa, φb, q12))
            end
        end
        close(f_traj)

        # Aggregate and write summary + per-trial
        n_cp = length(CHECKPOINTS)
        for (ci, cp) in enumerate(CHECKPOINTS)
            phi_a_all = [results[d].phi_a[ci] for d in 1:N_DISORDER]
            phi_b_all = [results[d].phi_b[ci] for d in 1:N_DISORDER]
            q_all     = [results[d].q[ci] for d in 1:N_DISORDER]
            pm_all    = [results[d].phimax[ci] for d in 1:N_DISORDER]
            φ_all     = vcat(phi_a_all, phi_b_all)
            φ_mean    = mean(φ_all)
            q_mean    = mean(q_all)
            qmφ2      = mean(q_all .- ((phi_a_all .+ phi_b_all) ./ 2).^2)

            pm_mean = ci == n_cp ? mean(pm_all) : NaN
            pm_std  = ci == n_cp ? std(pm_all) : NaN

            write(f_summary, @sprintf("%.3f,%.3f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                  α, T, cp, φ_mean, std(φ_all), q_mean, std(q_all),
                  isnan(pm_mean) ? 0.0 : pm_mean, isnan(pm_std) ? 0.0 : pm_std, qmφ2))

            @printf("  cp=%6d: φ=%.4f±%.4f  q=%.4f±%.4f  q-φ²=%+.5f\n",
                    cp, φ_mean, std(φ_all), q_mean, std(q_all), qmφ2)

            # Per-trial
            for d in 1:N_DISORDER
                write(f_trials, @sprintf("%.3f,%.3f,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                      α, T, cp, d, results[d].phi_a[ci], results[d].phi_b[ci],
                      results[d].q[ci], isnan(results[d].phimax[ci]) ? 0.0 : results[d].phimax[ci]))
            end
        end
    end

    close(f_summary)
    close(f_trials)

    println("\n" * "=" ^ 70)
    println("Output: v9_summary.csv, v9_per_trial.csv, v9_trajectory_*.csv")
    println("=" ^ 70)
end

main()

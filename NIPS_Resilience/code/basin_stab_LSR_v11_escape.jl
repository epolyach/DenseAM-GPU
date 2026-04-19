#=
GPU-Accelerated LSR Escape-Time Measurement (v11)
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v11_escape.jl
  julia basin_stab_LSR_v11_escape.jl --fresh

Goal: Measure the Kramers escape time from the retrieval basin by
  recording φ(t) trajectories at selected (α,T) probe points in the
  "resolvable" regime where τ_eff ~ 10³–10⁵ MC steps.

Design:
  - ONE (α,T) point at a time → all GPU memory for disorder samples
  - n_dis = 2000 disorder samples × 2 replicas = 4000 chains
  - Record φ every TRAJ_STRIDE steps → detect hop times
  - 2^18 = 262144 MC steps per trial
  - Sequential over probe points (~26 sec each on A6000)

Output per probe point:
  v11_trajectory_a{α}_T{T}.csv:  step, disorder, phi_a, phi_b, phimax_a
  v11_summary.csv:               α, T, N, M, n_dis, tau_1hop_median, ...

Probe points: α ∈ {0.18,0.20,0.22,0.24}, T ∈ {0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.80}
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ──────────────── Precision ────────────────
const F = Float32

# ──────────────── LSR Parameters ────────────────
const b_lsr     = F(2 + sqrt(2))
const PHI_MIN   = F(0.75)
const PHI_MAX   = F(1.0)

# ──────────────── Simulation Parameters ────────────────
const N_STEPS     = 2^18              # 262144 total MC steps
const TRAJ_STRIDE = 64               # record φ every 64 steps
const N_TRAJ      = N_STEPS ÷ TRAJ_STRIDE  # 4096 trajectory points
const N_DIS       = 2000             # disorder samples
const M_FIXED     = 20000            # fixed M for all α (keeps N reasonable)

const INF_ENERGY = F(1e30)

# Probe points: (α, T)
const PROBE_POINTS = [
    # α     T
    (0.18, 0.10), (0.18, 0.15), (0.18, 0.20), (0.18, 0.25),
    (0.18, 0.30), (0.18, 0.40), (0.18, 0.50), (0.18, 0.80),
    (0.20, 0.10), (0.20, 0.15), (0.20, 0.20), (0.20, 0.25),
    (0.20, 0.30), (0.20, 0.40), (0.20, 0.50), (0.20, 0.80),
    (0.22, 0.10), (0.22, 0.15), (0.22, 0.20), (0.22, 0.25),
    (0.22, 0.30), (0.22, 0.40), (0.22, 0.50), (0.22, 0.80),
    (0.24, 0.10), (0.24, 0.15), (0.24, 0.20), (0.24, 0.25),
    (0.24, 0.30), (0.24, 0.40), (0.24, 0.50), (0.24, 0.80),
]

# ──────────────── LSR Energy ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuMatrix{F},
                              patterns::CuMatrix{F}, overlap::CuMatrix{F},
                              Nf::F)
    Nb = Nf / b_lsr
    # overlap[M, n_chains] = patterns' * x
    mul!(overlap, patterns', x)
    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = vec(sum(overlap, dims=1))
    E .= @. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY)
    return nothing
end

# ──────────────── MC Step ────────────────
function mc_step!(x::CuMatrix{F}, xp::CuMatrix{F},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuMatrix{F}, ov::CuMatrix{F},
                  β::F, Nf::F, σ::F)
    N, n_chains = size(x)
    randn!(xp)
    @. xp = x + σ * xp
    # Project onto sphere
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    ra = CUDA.rand(F, n_chains)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-β * (Ep - E)))
    acc2 = reshape(acc, 1, n_chains)
    @. x = ifelse(acc2, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Compute φ and φ_max_other ────────────────
function compute_phi!(phi_out::CuVector{F}, phimax_out::CuVector{F},
                      x::CuMatrix{F}, target::CuVector{F},
                      patterns::CuMatrix{F}, overlap::CuMatrix{F}, Nf::F)
    N, n_chains = size(x)
    # φ with target
    phi_out .= vec(target' * x) ./ Nf

    # φ_max_other: max over non-target patterns
    mul!(overlap, patterns', x)
    @. overlap = overlap / Nf
    overlap[1, :] .= F(-1e30)  # mask target
    phimax_out .= vec(maximum(overlap, dims=1))
    return nothing
end

# ──────────────── Initialize near target ────────────────
function initialize_near!(x::Matrix{F}, target::Vector{F}, N::Int)
    n_chains = size(x, 2)
    for j in 1:n_chains
        phi_init = PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F)
        x_perp = randn(F, N)
        ov = dot(target, x_perp) / N
        x_perp .-= ov .* target
        x_perp ./= norm(x_perp)
        x[:, j] .= phi_init .* target .+ sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
    end
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    fresh = "--fresh" in ARGS

    println("=" ^ 70)
    println("LSR Escape-Time Measurement – GPU v11")
    println("  Probe points: $(length(PROBE_POINTS))")
    println("  MC steps: $N_STEPS (record every $TRAJ_STRIDE → $N_TRAJ points)")
    println("  Disorder samples: $N_DIS × 2 replicas = $(2*N_DIS) chains")
    println("  M = $M_FIXED (fixed)")
    println("=" ^ 70)

    # Summary file
    summary_file = "v11_summary.csv"
    if fresh || !isfile(summary_file)
        open(summary_file, "w") do f
            write(f, "alpha,T,N,M,n_dis,phi_mean_final,phi_std_final,frac_escaped,tau_1hop_median\n")
        end
    end

    for (pi, (α, T)) in enumerate(PROBE_POINTS)
        N = max(round(Int, log(M_FIXED) / α), 2)
        M = M_FIXED
        Nf = F(N)
        β = F(1 / T)
        σ = F(2.4 * T / sqrt(Float64(N)))
        n_chains = 2 * N_DIS  # 2 replicas

        @printf("\n── Point %d/%d: α=%.2f, T=%.2f (N=%d, M=%d, chains=%d) ──\n",
                pi, length(PROBE_POINTS), α, T, N, M, n_chains)

        # Check GPU memory
        mem_patterns = N * M * 4 / 1e9
        mem_overlap  = M * n_chains * 4 / 1e9
        mem_states   = N * n_chains * 2 * 4 / 1e9  # x and xp
        mem_traj     = N_TRAJ * n_chains * 2 * 4 / 1e9  # phi and phimax trajectories
        mem_total    = mem_patterns + mem_overlap + mem_states + mem_traj
        @printf("  Memory: patterns %.1f + overlap %.1f + states %.1f + traj %.1f = %.1f GB\n",
                mem_patterns, mem_overlap, mem_states, mem_traj, mem_total)

        # Generate patterns on CPU
        Random.seed!(42000 + pi)
        pat_cpu = randn(F, N, M)
        for j in 1:M
            c = @view pat_cpu[:, j]
            c .*= sqrt(Nf) / norm(c)
        end
        target_cpu = pat_cpu[:, 1]

        # Initialize replicas on CPU
        xa_cpu = zeros(F, N, N_DIS)
        xb_cpu = zeros(F, N, N_DIS)
        Random.seed!(100000 + pi)
        initialize_near!(xa_cpu, target_cpu, N)
        Random.seed!(200000 + pi)
        initialize_near!(xb_cpu, target_cpu, N)

        # Interleave replicas: x[:, 1:N_DIS] = A, x[:, N_DIS+1:end] = B
        x_cpu = hcat(xa_cpu, xb_cpu)

        # Transfer to GPU
        pat_g = CuArray(pat_cpu)
        x_g   = CuArray(x_cpu)
        xp_g  = similar(x_g)
        target_g = CuArray(target_cpu)

        ov_g  = CUDA.zeros(F, M, n_chains)
        E_g   = CUDA.zeros(F, n_chains)
        Ep_g  = CUDA.zeros(F, n_chains)
        phi_g    = CUDA.zeros(F, n_chains)
        phimax_g = CUDA.zeros(F, n_chains)

        # Trajectory storage (CPU, filled incrementally)
        traj_phi    = zeros(Float32, N_TRAJ, n_chains)
        traj_phimax = zeros(Float32, N_TRAJ, n_chains)

        # Compute initial energy
        compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

        # ── MC loop ──
        traj_idx = 0
        t_start = time()

        prog = Progress(N_STEPS, desc="  MC: ")
        for step in 1:N_STEPS
            mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ)

            if step % TRAJ_STRIDE == 0
                traj_idx += 1
                compute_phi!(phi_g, phimax_g, x_g, target_g, pat_g, ov_g, Nf)
                traj_phi[traj_idx, :]    .= Array(phi_g)
                traj_phimax[traj_idx, :] .= Array(phimax_g)
            end

            next!(prog)
        end
        t_elapsed = time() - t_start
        @printf("  Done: %.1f s (%.3f ms/step)\n", t_elapsed, 1000*t_elapsed/N_STEPS)

        # ── Save trajectory ──
        traj_file = @sprintf("v11_trajectory_a%.2f_T%.2f.csv", α, T)
        open(traj_file, "w") do f
            write(f, "step,disorder,phi_a,phi_b,phimax_a,phimax_b\n")
            # Subsample: save every 4th trajectory point to keep files manageable
            for ti in 1:4:N_TRAJ
                step = ti * TRAJ_STRIDE
                for d in 1:N_DIS
                    @printf(f, "%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                            step, d,
                            traj_phi[ti, d], traj_phi[ti, d + N_DIS],
                            traj_phimax[ti, d], traj_phimax[ti, d + N_DIS])
                end
            end
        end
        @printf("  Saved: %s\n", traj_file)

        # ── Analyze: escape fraction and median hop time ──
        # Define "escaped" as φ < 0.86 (below retrieval peak)
        phi_final_a = traj_phi[end, 1:N_DIS]
        phi_final_b = traj_phi[end, N_DIS+1:end]
        phi_final = vcat(phi_final_a, phi_final_b)

        frac_escaped = mean(phi_final .< 0.86)
        phi_mean = mean(phi_final)
        phi_std  = std(phi_final)

        # Estimate median first-hop time: first time φ drops below 0.86
        hop_times = Float64[]
        for j in 1:n_chains
            for ti in 1:N_TRAJ
                if traj_phi[ti, j] < 0.86
                    push!(hop_times, ti * TRAJ_STRIDE)
                    break
                end
            end
        end
        tau_median = isempty(hop_times) ? Inf : median(hop_times)

        @printf("  φ_final: mean=%.3f ± %.3f, escaped=%.1f%%, τ_1hop_median=%.0f\n",
                phi_mean, phi_std, 100*frac_escaped, tau_median)

        # Append to summary
        open(summary_file, "a") do f
            @printf(f, "%.2f,%.2f,%d,%d,%d,%.4f,%.4f,%.4f,%.1f\n",
                    α, T, N, M, N_DIS, phi_mean, phi_std, frac_escaped, tau_median)
        end

        # Free GPU memory
        CUDA.unsafe_free!(pat_g)
        CUDA.unsafe_free!(x_g)
        CUDA.unsafe_free!(xp_g)
        CUDA.unsafe_free!(ov_g)
        CUDA.unsafe_free!(E_g)
        CUDA.unsafe_free!(Ep_g)
        CUDA.unsafe_free!(phi_g)
        CUDA.unsafe_free!(phimax_g)
        GC.gc()
        CUDA.reclaim()
    end

    println("\n" * "=" ^ 70)
    println("All probe points complete. Summary: $summary_file")
    println("=" ^ 70)
end

main()

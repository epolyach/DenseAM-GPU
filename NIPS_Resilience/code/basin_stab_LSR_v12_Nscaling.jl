#=
GPU-Accelerated LSR N-Scaling Test (v12)
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v12_Nscaling.jl
  julia basin_stab_LSR_v12_Nscaling.jl --fresh

Goal: Test the N-dependence of the Kramers escape time at FIXED α=0.20
  by varying M (which changes N = ⌊ln M / α⌋ while keeping φ_{1,max} fixed).

  This directly tests τ ∝ N·τ_rel·exp(c·ΔF/T) with ΔF/T ∝ N.

Design:
  Same as v11 but with variable M per probe point.
  - M ∈ {20k, 100k, 500k, 2M} → N ∈ {49, 57, 65, 72}
  - α = 0.20 fixed → φ_{1,max} = 0.574 at all N
  - T ∈ {0.15, 0.25, 0.40, 0.80} (4 temperatures)
  - n_dis reduced for large M to fit GPU memory

Output:
  v12_trajectory_a{α}_M{M}_T{T}.csv
  v12_summary.csv
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ──────────────── Precision ────────────────
const F = Float16    # halves memory bandwidth → 2× faster MC steps

# ──────────────── LSR Parameters ────────────────
const b_lsr     = F(2 + sqrt(2))
const PHI_MIN   = F(0.75)
const PHI_MAX   = F(1.0)
const INF_ENERGY = F(1e30)

# ──────────────── Simulation Parameters ────────────────
const N_STEPS     = 2^18              # 262144 total MC steps
const TRAJ_STRIDE = 64               # record φ every 64 steps
const N_TRAJ      = N_STEPS ÷ TRAJ_STRIDE  # 4096 trajectory points
const M_DEFAULT   = 20000            # default M
const N_DIS_DEFAULT = 2000           # default disorder samples

# ──────────────── N-scaling probe points ────────────────
# Fixed α=0.20, variable M → variable N = ⌊ln(M)/0.20⌋
# (α, T, M, n_dis) — n_dis reduced for large M to fit ~40GB GPU
const ALPHA_FIXED = 0.20
const T_VALUES = [0.15, 0.25, 0.40, 0.80]
const M_N_DIS = [
    (  20_000, 2000),   # N=49, baseline (already in v11, included for self-consistency)
    ( 100_000,  800),   # N=57
    ( 500_000,  300),   # N=65
    (2_000_000, 100),   # N=72
]

const PROBE_POINTS = Tuple[]
for (M_val, nd) in M_N_DIS
    for T_val in T_VALUES
        push!(PROBE_POINTS, (ALPHA_FIXED, T_val, M_val, nd))
    end
end

# Helper to extract (α, T, M, n_dis) from a probe point tuple
function parse_probe(pt)
    if length(pt) == 2
        return (pt[1], pt[2], M_DEFAULT, N_DIS_DEFAULT)
    elseif length(pt) == 4
        return (pt[1], pt[2], pt[3], pt[4])
    else
        error("Probe point must be (α, T) or (α, T, M, n_dis)")
    end
end

# ──────────────── LSR Energy (batched over disorder samples) ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    Nb = Nf / b_lsr
    # overlap[M, n_rep, n_dis] = patterns' * x  (batched GEMM)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1)
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

# ──────────────── MC Step (batched) ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::F, Nf::F, σ::F, ra::CuVector{F})
    randn!(xp)
    @. xp = x + σ * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-β * (Ep - E)))
    n_rep = size(x, 2)
    n_dis = size(x, 3)
    a3 = reshape(acc, 1, n_rep, n_dis)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Compute φ with target (batched) ────────────────
function compute_phi!(phi_out::CuVector{F}, x::CuArray{F,3},
                      targets::CuArray{F,3}, Nf::F)
    # targets[N, 1, n_dis], x[N, n_rep, n_dis]
    # φ = sum(target .* x, dims=1) / N
    phi_out .= vec(sum(targets .* x, dims=1)) ./ Nf
    return nothing
end

# ──────────────── Compute φ_max_other (batched) ────────────────
function compute_phimax!(phimax_out::CuVector{F}, x::CuArray{F,3},
                         patterns::CuArray{F,3}, overlap::CuArray{F,3},
                         Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = overlap / Nf
    overlap[1, :, :] .= F(-1e30)  # mask target (pattern 1)
    mx = maximum(overlap, dims=1)
    phimax_out .= vec(mx)
    return nothing
end

# ──────────────── Initialize near target (CPU) ────────────────
function initialize_near!(x::Array{F,3}, targets::Array{F,3}, N::Int)
    n_rep, n_dis = size(x, 2), size(x, 3)
    for d in 1:n_dis
        target = @view targets[:, 1, d]
        for r in 1:n_rep
            phi_init = PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F)
            x_perp = randn(F, N)
            ov = dot(target, x_perp) / N
            x_perp .-= ov .* target
            x_perp ./= norm(x_perp)
            x[:, r, d] .= phi_init .* target .+ sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
        end
    end
end


# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    fresh = "--fresh" in ARGS

    println("=" ^ 70)
    println("LSR N-Scaling Test – GPU v12 (fixed α, variable M)")
    println("  Probe points: $(length(PROBE_POINTS))")
    println("  MC steps: $N_STEPS (record every $TRAJ_STRIDE → $N_TRAJ points)")
    println("  α = $ALPHA_FIXED (fixed)")
    println("  M values: $(first.(M_N_DIS))")
    println("  T values: $T_VALUES")
    println("=" ^ 70)

    summary_file = "v12_summary.csv"
    if fresh || !isfile(summary_file)
        open(summary_file, "w") do f
            write(f, "alpha,T,N,M,n_dis,phi_mean_final,phi_std_final,frac_escaped,tau_1hop_median\n")
        end
    end

    for (pi, pt) in enumerate(PROBE_POINTS)
        α, T, M, n_dis = parse_probe(pt)
        N = max(round(Int, log(M) / α), 2)
        Nf = F(N)
        β = F(1 / T)
        σ = F(2.4 * T / sqrt(Float64(N)))
        n_rep = 2  # two replicas per disorder sample
        n_chains = n_rep * n_dis

        @printf("\n── Point %d/%d: α=%.2f, T=%.2f (N=%d, M=%d, n_dis=%d, chains=%d) ──\n",
                pi, length(PROBE_POINTS), α, T, N, M, n_dis, n_chains)

        mem_gb = (N * M * n_dis + N * n_rep * n_dis * 2 + M * n_rep * n_dis) * 4 / 1e9
        @printf("  GPU memory: %.1f GB\n", mem_gb)

        # ── Generate patterns on CPU (independent per disorder sample) ──
        Random.seed!(42000 + pi)
        pat_cpu = randn(F, N, M, n_dis)
        for d in 1:n_dis
            for j in 1:M
                c = @view pat_cpu[:, j, d]
                c .*= sqrt(Nf) / norm(c)
            end
        end
        # Extract targets: first pattern of each disorder sample
        tgt_cpu = reshape(pat_cpu[:, 1, :], N, 1, n_dis)

        # ── Initialize replicas on CPU ──
        x_cpu = zeros(F, N, n_rep, n_dis)
        Random.seed!(100000 + pi)
        initialize_near!(x_cpu, tgt_cpu, N)

        # ── Transfer to GPU ──
        pat_g = CuArray(pat_cpu)
        tgt_g = CuArray(tgt_cpu)
        x_g   = CuArray(x_cpu)
        xp_g  = similar(x_g)

        ov_g  = CUDA.zeros(F, M, n_rep, n_dis)
        E_g   = CUDA.zeros(F, n_chains)
        Ep_g  = CUDA.zeros(F, n_chains)
        ra_g  = CUDA.zeros(F, n_chains)
        phi_g    = CUDA.zeros(F, n_chains)
        phimax_g = CUDA.zeros(F, n_chains)

        # Free CPU memory
        pat_cpu = nothing; x_cpu = nothing; tgt_cpu = nothing
        GC.gc()

        # Trajectory storage (CPU)
        traj_phi    = zeros(Float32, N_TRAJ, n_chains)
        traj_phimax = zeros(Float32, N_TRAJ, n_chains)

        # Compute initial energy
        compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

        # ── MC loop ──
        traj_idx = 0
        t_start = time()

        prog = Progress(N_STEPS, desc="  MC: ")
        for step in 1:N_STEPS
            mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ, ra_g)

            if step % TRAJ_STRIDE == 0
                traj_idx += 1
                compute_phi!(phi_g, x_g, tgt_g, Nf)
                compute_phimax!(phimax_g, x_g, pat_g, ov_g, Nf)
                traj_phi[traj_idx, :]    .= Array(phi_g)
                traj_phimax[traj_idx, :] .= Array(phimax_g)
            end

            next!(prog)
        end
        t_elapsed = time() - t_start
        @printf("  Done: %.1f s (%.3f ms/step)\n", t_elapsed, 1000*t_elapsed/N_STEPS)

        # ── Save trajectory (subsampled) ──
        traj_file = @sprintf("v12_trajectory_a%.2f_M%d_T%.2f.csv", α, M, T)
        open(traj_file, "w") do f
            write(f, "step,disorder,phi_a,phi_b,phimax_a,phimax_b\n")
            for ti in 1:4:N_TRAJ  # every 4th point
                step = ti * TRAJ_STRIDE
                for d in 1:min(n_dis, 500)  # cap CSV size: first 500 disorder samples
                    ja = (d - 1) * n_rep + 1  # replica A index
                    jb = (d - 1) * n_rep + 2  # replica B index
                    @printf(f, "%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                            step, d,
                            traj_phi[ti, ja], traj_phi[ti, jb],
                            traj_phimax[ti, ja], traj_phimax[ti, jb])
                end
            end
        end
        @printf("  Saved: %s\n", traj_file)

        # ── Analyze: escape fraction and median hop time ──
        phi_final = [traj_phi[end, (d-1)*n_rep + r] for d in 1:n_dis for r in 1:n_rep]
        frac_escaped = mean(phi_final .< 0.86)
        phi_mean = mean(phi_final)
        phi_std  = std(phi_final)

        # Median first-hop time: first time φ drops below 0.86
        hop_times = Float64[]
        for d in 1:n_dis, r in 1:n_rep
            j = (d - 1) * n_rep + r
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
                    α, T, N, M, n_dis, phi_mean, phi_std, frac_escaped, tau_median)
        end

        # Free GPU memory
        CUDA.unsafe_free!(pat_g)
        CUDA.unsafe_free!(tgt_g)
        CUDA.unsafe_free!(x_g)
        CUDA.unsafe_free!(xp_g)
        CUDA.unsafe_free!(ov_g)
        CUDA.unsafe_free!(E_g)
        CUDA.unsafe_free!(Ep_g)
        CUDA.unsafe_free!(ra_g)
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

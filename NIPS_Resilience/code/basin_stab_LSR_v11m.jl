#=
GPU-Accelerated LSR Escape-Time Measurement (v11m) — Self-contained
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v11m.jl
  julia basin_stab_LSR_v11m.jl --fresh    # overwrite summary, rerun all

Runs all probe points sequentially. Skips points already in the summary
CSV, so it can be stopped and resumed safely.

Probe points: all combinations of
  α ∈ {0.18, 0.20, 0.22, 0.24, 0.26, 0.28}
  T ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80}
  M ∈ {20_000}   (default; add more below for N-scaling)

For N-scaling (v12-style), add entries to M_VALUES below, e.g.
  M ∈ {20_000, 100_000, 500_000, 2_000_000}
with appropriate n_dis (see GPU_MEM_TARGET_GB).

Output:
  v11m_summary.csv                         — one row per (α, T, M) point
  v11m_trajectory_a{α}_M{M}_T{T}.csv      — φ(t) trajectories
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ══════════════════════════════════════════════════════════════════════
#                         USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

const ALPHA_VALUES = [0.18, 0.20, 0.22, 0.24, 0.26, 0.28]
const T_VALUES     = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80]
const M_VALUES     = [50_000, 100_000, 500_000, 2_000_000]   # N-scaling (M=20k baseline already in v11)

const GPU_MEM_TARGET_GB = 35.0   # target GPU memory usage (~80% of A6000's 48 GB)
const N_DIS_MAX         = 2000   # cap on disorder samples (even if memory allows more)
const N_STEPS           = 2^18   # 262144 total MC steps
const TRAJ_STRIDE       = 64     # record φ every 64 steps
const SUMMARY_FILE      = "v11m_summary.csv"

# ══════════════════════════════════════════════════════════════════════

const F = Float16
const b_lsr      = F(2 + sqrt(2))
const PHI_MIN    = F(0.75)
const PHI_MAX    = F(1.0)
const INF_ENERGY = F(1e30)
const N_TRAJ     = N_STEPS ÷ TRAJ_STRIDE

# ──────────────── Auto-size n_dis to fit GPU memory ────────────────
function auto_n_dis(N::Int, M::Int)
    # Memory per disorder sample (Float16 = 2 bytes):
    #   patterns:  N × M × 2
    #   replicas:  N × 2 × 2     (x and xp)
    #   overlap:   M × 2 × 2
    #   scalars:   ~6 vectors of length 2  (negligible)
    bytes_per_dis = 2 * (N * M + 2 * N * 2 + M * 2 * 2)
    n = min(floor(Int, GPU_MEM_TARGET_GB * 1e9 / bytes_per_dis), N_DIS_MAX)
    return max(n, 1)
end

# ──────────────── LSR Energy (batched) ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    Nb = Nf / b_lsr
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
    phi_out .= vec(sum(targets .* x, dims=1)) ./ Nf
    return nothing
end

# ──────────────── Compute φ_max_other (batched) ────────────────
function compute_phimax!(phimax_out::CuVector{F}, x::CuArray{F,3},
                         patterns::CuArray{F,3}, overlap::CuArray{F,3},
                         Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = overlap / Nf
    overlap[1, :, :] .= F(-1e30)
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

# ──────────────── Check if point is already done ────────────────
function already_done(summary_file, α, T, M)
    !isfile(summary_file) && return false
    for line in readlines(summary_file)[2:end]
        f = split(line, ",")
        length(f) < 4 && continue
        if isapprox(parse(Float64,f[1]), α; atol=0.001) &&
           isapprox(parse(Float64,f[2]), T; atol=0.001) &&
           parse(Int, f[4]) == M
            return true
        end
    end
    return false
end

# ──────────────── Run one probe point ────────────────
function run_point!(α, T, M, n_dis, summary_file)
    N = max(round(Int, log(M) / α), 2)
    Nf = F(N)
    β = F(1 / T)
    σ = F(2.4 * T / sqrt(Float64(N)))
    n_rep = 2
    n_chains = n_rep * n_dis

    @printf("  α=%.2f, T=%.2f, M=%d, N=%d, n_dis=%d, chains=%d\n",
            α, T, M, N, n_dis, n_chains)

    mem_gb = (N * M * n_dis + N * n_rep * n_dis * 2 + M * n_rep * n_dis) * 2 / 1e9
    @printf("  GPU memory estimate: %.1f GB\n", mem_gb)

    # Generate patterns on CPU
    Random.seed!(hash((α, T, M)))
    pat_cpu = randn(F, N, M, n_dis)
    for d in 1:n_dis, j in 1:M
        c = @view pat_cpu[:, j, d]
        c .*= sqrt(Nf) / norm(c)
    end
    tgt_cpu = reshape(pat_cpu[:, 1, :], N, 1, n_dis)

    # Initialize replicas
    x_cpu = zeros(F, N, n_rep, n_dis)
    Random.seed!(hash((α, T, M, :init)))
    initialize_near!(x_cpu, tgt_cpu, N)

    # Transfer to GPU
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

    pat_cpu = nothing; x_cpu = nothing; tgt_cpu = nothing
    GC.gc()

    # Trajectory storage
    traj_phi    = zeros(Float32, N_TRAJ, n_chains)
    traj_phimax = zeros(Float32, N_TRAJ, n_chains)

    compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

    # MC loop
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

    # Save trajectory (subsampled)
    traj_file = @sprintf("v11m_trajectory_a%.2f_M%d_T%.2f.csv", α, M, T)
    open(traj_file, "w") do f
        write(f, "step,disorder,phi_a,phi_b,phimax_a,phimax_b\n")
        for ti in 1:4:N_TRAJ
            step = ti * TRAJ_STRIDE
            for d in 1:min(n_dis, 500)
                ja = (d - 1) * n_rep + 1
                jb = (d - 1) * n_rep + 2
                @printf(f, "%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                        step, d,
                        traj_phi[ti, ja], traj_phi[ti, jb],
                        traj_phimax[ti, ja], traj_phimax[ti, jb])
            end
        end
    end
    @printf("  Saved: %s\n", traj_file)

    # Analyze
    phi_final = [traj_phi[end, (d-1)*n_rep + r] for d in 1:n_dis for r in 1:n_rep]
    frac_escaped = mean(phi_final .< 0.86)
    phi_mean = mean(phi_final)
    phi_std  = std(phi_final)

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

    # Free GPU
    CUDA.unsafe_free!(pat_g); CUDA.unsafe_free!(tgt_g)
    CUDA.unsafe_free!(x_g);   CUDA.unsafe_free!(xp_g)
    CUDA.unsafe_free!(ov_g);  CUDA.unsafe_free!(E_g)
    CUDA.unsafe_free!(Ep_g);  CUDA.unsafe_free!(ra_g)
    CUDA.unsafe_free!(phi_g); CUDA.unsafe_free!(phimax_g)
    GC.gc(); CUDA.reclaim()
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    fresh = "--fresh" in ARGS

    # Build probe list: all combinations of (α, T, M)
    probes = Tuple{Float64,Float64,Int}[]
    for α in ALPHA_VALUES, T_val in T_VALUES, M_val in M_VALUES
        push!(probes, (α, T_val, M_val))
    end

    println("=" ^ 70)
    println("LSR Escape-Time Measurement – GPU v11m (self-contained)")
    println("  α values: $ALPHA_VALUES")
    println("  T values: $T_VALUES")
    println("  M values: $M_VALUES")
    println("  Total probe points: $(length(probes))")
    println("  MC steps: $N_STEPS (record every $TRAJ_STRIDE → $N_TRAJ points)")
    println("  GPU memory target: $(GPU_MEM_TARGET_GB) GB")
    println("  n_dis cap: $N_DIS_MAX")
    println("=" ^ 70)

    if fresh || !isfile(SUMMARY_FILE)
        open(SUMMARY_FILE, "w") do f
            write(f, "alpha,T,N,M,n_dis,phi_mean_final,phi_std_final,frac_escaped,tau_1hop_median\n")
        end
    end

    n_done = 0; n_skip = 0
    for (pi, (α, T_val, M_val)) in enumerate(probes)
        N = max(round(Int, log(M_val) / α), 2)
        n_dis = auto_n_dis(N, M_val)

        if !fresh && already_done(SUMMARY_FILE, α, T_val, M_val)
            n_skip += 1
            @printf("── Point %d/%d: α=%.2f, T=%.2f, M=%d — SKIP (already done) ──\n",
                    pi, length(probes), α, T_val, M_val)
            continue
        end

        @printf("\n── Point %d/%d ──\n", pi, length(probes))
        run_point!(α, T_val, M_val, n_dis, SUMMARY_FILE)
        n_done += 1
    end

    println("\n" * "=" ^ 70)
    @printf("Complete. %d points run, %d skipped. Summary: %s\n",
            n_done, n_skip, SUMMARY_FILE)
    println("=" ^ 70)
end

main()

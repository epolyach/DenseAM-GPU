#=
v14 — Clean Kramers escape-time measurement
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v14_tau.jl
  julia basin_stab_LSR_v14_tau.jl --fresh

Design:
  - Initialize at φ=1 (x = √N · ξ¹) — no initialization artifact
  - Adaptive stride and run length per (α,T) point
  - Record P_esc(t) = fraction escaped by time t
  - Compute τ(t) = -t/ln(1-P_esc(t)) — should plateau at τ_Kramers
  - Escape threshold: (φ_eq(T) + φ_cen(T))/2

  Focus on the "resolvable" regime: τ ~ 1k–100k MC steps
  (fast enough to measure, slow enough for clean exponential)

Output:
  v14_summary.csv:         α, T, N, M, n_dis, tau_plateau, P_esc_final
  v14_Pesc_a{α}_T{T}.csv:  t, P_esc(t), tau(t)   (the survival curve)
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ══════════════════════════════════════════════════════════════════════
#                         CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

const GPU_MEM_TARGET_GB = 35.0
const N_DIS_MAX         = 4000
const M_PAT             = 20000
const SUMMARY_FILE      = "v14_summary.csv"

# Probe points: (α, T, T_run, stride)
# Selected for τ ~ 1k–100k (fast computation, clean measurement)
# τ estimates from v11 survival analysis (cleaned)
const PROBE_POINTS = [
    # α     T     T_run    stride   τ_est
    (0.20, 0.25, 500_000,   500),  # τ~900k — long but α=0.20 is important
    (0.20, 0.30, 500_000,   500),  # τ~80k
    (0.20, 0.40, 100_000,    64),  # τ~4k
    (0.20, 0.50,  50_000,    16),  # τ~1k
    (0.20, 0.80,  10_000,     4),  # τ~300
    (0.22, 0.20, 500_000,   500),  # τ~680k
    (0.22, 0.25, 500_000,   250),  # τ~290k
    (0.22, 0.30, 300_000,   250),  # τ~47k
    (0.22, 0.40, 100_000,    32),  # τ~3k
    (0.22, 0.50,  30_000,     8),  # τ~1k
    (0.22, 0.80,  10_000,     4),  # τ~300
    (0.24, 0.15, 500_000,   500),  # τ~340k
    (0.24, 0.20, 500_000,   250),  # τ~160k
    (0.24, 0.25, 300_000,   250),  # τ~83k
    (0.24, 0.30, 200_000,   128),  # τ~22k
    (0.24, 0.40,  50_000,    16),  # τ~3k
    (0.24, 0.50,  20_000,     8),  # τ~900
    (0.24, 0.80,  10_000,     4),  # τ~300
    (0.26, 0.20, 300_000,   250),  # τ~?
    (0.26, 0.25, 200_000,   128),  # τ~?
    (0.26, 0.30, 100_000,    64),  # τ~?
    (0.26, 0.40,  30_000,    16),  # τ~?
    (0.26, 0.50,  10_000,     4),  # τ~?
    (0.28, 0.15, 200_000,   128),  # τ~?
    (0.28, 0.20, 100_000,    64),  # τ~?
    (0.28, 0.25,  50_000,    32),  # τ~?
    (0.28, 0.30,  30_000,    16),  # τ~?
    (0.28, 0.40,  10_000,     4),  # τ~?
    (0.28, 0.50,  10_000,     4),  # τ~?
]

# ══════════════════════════════════════════════════════════════════════

const F = Float16
const b_lsr      = F(2 + sqrt(2))
const PHI_C      = Float64((Float64(b_lsr)-1)/Float64(b_lsr))
const INF_ENERGY = F(1e30)

function auto_n_dis(N::Int, M::Int)
    bytes_per_dis = 2 * (N * M + N * 2 + M * 2)
    return min(floor(Int, GPU_MEM_TARGET_GB * 1e9 / bytes_per_dis), N_DIS_MAX)
end

# ──────────────── LSR theory (CPU) ────────────────
function φ_eq_LSR(T)
    b = Float64(b_lsr)
    pc = PHI_C
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b + b*φ
        D ≤ 1e-10 && (φ = pc + 0.005; continue)
        f = (1 - φ^2) - T*φ*D
        fp = -2φ - T*(D + b*φ)
        φ = clamp(φ - f/fp, pc + 1e-8, 1 - 1e-8)
    end
    return φ
end

function φ_cen_eq(T, φ_1mu)
    b = Float64(b_lsr)
    pc = PHI_C
    φ_max = sqrt((1 + φ_1mu) / 2) - 1e-6
    φ = φ_max * 0.95
    for _ in 1:300
        D = 1 - b + b*φ
        D ≤ 1e-10 && (φ = pc + 0.005; continue)
        f = (1 + φ_1mu - 2φ^2) - 2T*φ*D
        fp = -4φ - 2T*(D + b*φ)
        φ = clamp(φ - f/fp, pc + 1e-8, φ_max)
    end
    return φ
end

# ──────────────── GPU kernels (same as v11m) ────────────────
function compute_energy_lsr!(E, x, pat, ov, Nf)
    Nb = Nf / b_lsr
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), pat, x, zero(F), ov)
    @. ov = max(zero(F), one(F) - b_lsr + b_lsr * ov / Nf)
    s = sum(ov, dims=1)
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

function mc_step!(x, xp, E, Ep, pat, ov, β, Nf, σ, ra)
    randn!(xp)
    @. xp = x + σ * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm
    compute_energy_lsr!(Ep, xp, pat, ov, Nf)
    rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-β * (Ep - E)))
    a3 = reshape(acc, 1, size(x,2), size(x,3))
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
end

function compute_phi!(phi_out, x, tgt, Nf)
    phi_out .= vec(sum(tgt .* x, dims=1)) ./ Nf
end

# ──────────────── Check if done ────────────────
function already_done(sf, α, T)
    !isfile(sf) && return false
    for line in readlines(sf)[2:end]
        f = split(line, ",")
        length(f) < 3 && continue
        isapprox(parse(Float64,f[1]), α; atol=0.001) &&
        isapprox(parse(Float64,f[2]), T; atol=0.001) && return true
    end
    return false
end

# ──────────────── Run one point ────────────────
function run_point!(α, T, T_run, stride, n_dis)
    N = max(round(Int, log(M_PAT) / α), 2)
    M = M_PAT
    Nf = F(N)
    β = F(1 / T)
    σ = F(2.4 * T / sqrt(Float64(N)))
    n_rep = 1
    n_chains = n_rep * n_dis
    n_record = T_run ÷ stride

    # Compute escape threshold
    φ1max = sqrt(1 - exp(-2α))
    φeq = φ_eq_LSR(T)
    φcen = φ_cen_eq(T, φ1max)
    thresh = F((φeq + φcen) / 2)

    @printf("  α=%.2f, T=%.2f, N=%d, n_dis=%d, T_run=%d, stride=%d\n", α, T, N, n_dis, T_run, stride)
    @printf("  φ_eq=%.4f, φ_cen=%.4f, threshold=%.4f\n", φeq, φcen, Float64(thresh))

    # Generate patterns on CPU
    Random.seed!(hash((α, T, M, :v14)))
    pat_cpu = randn(F, N, M, n_dis)
    for d in 1:n_dis, j in 1:M
        c = @view pat_cpu[:, j, d]
        c .*= sqrt(Nf) / norm(c)
    end
    tgt_cpu = reshape(pat_cpu[:, 1, :], N, 1, n_dis)

    # Initialize at φ=1: x = √N · ξ¹ (exact target)
    x_cpu = zeros(F, N, n_rep, n_dis)
    for d in 1:n_dis
        x_cpu[:, 1, d] .= tgt_cpu[:, 1, d]  # x = ξ¹ (already normalized to √N)
    end

    # Transfer to GPU
    pat_g = CuArray(pat_cpu)
    tgt_g = CuArray(tgt_cpu)
    x_g   = CuArray(x_cpu)
    xp_g  = similar(x_g)
    ov_g  = CUDA.zeros(F, M, n_rep, n_dis)
    E_g   = CUDA.zeros(F, n_chains)
    Ep_g  = CUDA.zeros(F, n_chains)
    ra_g  = CUDA.zeros(F, n_chains)
    phi_g = CUDA.zeros(F, n_chains)

    pat_cpu = nothing; x_cpu = nothing; tgt_cpu = nothing
    GC.gc()

    compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

    # Track escape: has_escaped[j] = true once φ drops below threshold
    # first_escape_step[j] = step when escape happened (0 if not yet)
    has_escaped = CUDA.zeros(Int32, n_chains)  # 0=no, 1=yes
    escape_count = zeros(Int, n_record)  # cumulative escapes at each recorded step

    t_start = time()
    rec_idx = 0
    for step in 1:T_run
        mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ, ra_g)

        if step % stride == 0
            rec_idx += 1
            compute_phi!(phi_g, x_g, tgt_g, Nf)
            # Mark newly escaped trials
            newly = @. (phi_g < thresh) & (has_escaped == Int32(0))
            @. has_escaped = ifelse(newly, Int32(1), has_escaped)
            escape_count[rec_idx] = sum(Array(has_escaped))
        end
    end
    CUDA.synchronize()
    t_elapsed = time() - t_start
    @printf("  Done: %.1f s (%.3f ms/step)\n", t_elapsed, 1000*t_elapsed/T_run)

    # Compute P_esc(t) and τ(t)
    t_arr = [(i * stride) for i in 1:n_record]
    P_arr = escape_count ./ n_chains
    tau_arr = [P > 0 && P < 1 ? -t_arr[i] / log(1 - P) : NaN for (i, P) in enumerate(P_arr)]

    # Find plateau: τ(t) for t in [0.2×T_run, 0.8×T_run]
    plateau_mask = [0.2*T_run ≤ t_arr[i] ≤ 0.8*T_run && !isnan(tau_arr[i]) && tau_arr[i] > 0
                    for i in 1:n_record]
    if any(plateau_mask)
        tau_plateau = median(tau_arr[plateau_mask])
    else
        tau_plateau = NaN
    end

    P_final = P_arr[end]
    @printf("  P_esc_final=%.4f, τ_plateau=%.0f\n", P_final, tau_plateau)

    # Save P_esc(t) curve
    pesc_file = @sprintf("v14_Pesc_a%.2f_T%.2f.csv", α, T)
    open(pesc_file, "w") do f
        write(f, "step,P_esc,tau_t\n")
        for i in 1:n_record
            @printf(f, "%d,%.6f,%.1f\n", t_arr[i], P_arr[i], isnan(tau_arr[i]) ? -1.0 : tau_arr[i])
        end
    end
    @printf("  Saved: %s\n", pesc_file)

    # Append to summary
    open(SUMMARY_FILE, "a") do f
        @printf(f, "%.2f,%.2f,%d,%d,%d,%.1f,%.6f,%.4f,%.4f,%d,%d\n",
                α, T, N, M, n_dis, tau_plateau, P_final, φeq, Float64(thresh), T_run, stride)
    end

    # Free GPU
    for arr in [pat_g, tgt_g, x_g, xp_g, ov_g, E_g, Ep_g, ra_g, phi_g, has_escaped]
        CUDA.unsafe_free!(arr)
    end
    GC.gc(); CUDA.reclaim()
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")
    fresh = "--fresh" in ARGS

    println("=" ^ 70)
    println("v14 — Clean Kramers Escape-Time Measurement")
    println("  Initialize at φ=1, adaptive stride, escape threshold = (φ_eq+φ_cen)/2")
    println("  Points: $(length(PROBE_POINTS))")
    println("=" ^ 70)

    if fresh || !isfile(SUMMARY_FILE)
        open(SUMMARY_FILE, "w") do f
            write(f, "alpha,T,N,M,n_dis,tau_plateau,P_esc_final,phi_eq,threshold,T_run,stride\n")
        end
    end

    for (pi, (α, T, T_run, stride)) in enumerate(PROBE_POINTS)
        if !fresh && already_done(SUMMARY_FILE, α, T)
            @printf("── Point %d/%d: α=%.2f, T=%.2f — SKIP ──\n", pi, length(PROBE_POINTS), α, T)
            continue
        end

        N = max(round(Int, log(M_PAT) / α), 2)
        n_dis = auto_n_dis(N, M_PAT)

        @printf("\n── Point %d/%d ──\n", pi, length(PROBE_POINTS))
        run_point!(α, T, T_run, stride, n_dis)
    end

    println("\n" * "=" ^ 70)
    println("Complete. Summary: $SUMMARY_FILE")
    println("=" ^ 70)
end

main()

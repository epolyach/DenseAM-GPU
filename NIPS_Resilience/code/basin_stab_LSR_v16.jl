#=
v16 — Kramers escape measurement with confirmed barrier crossing
────────────────────────────────────────────────────────────────────────
Key improvements over v14:
  - Escape criterion: φ_μ ≥ φ_c for ANY μ≠1 (physical barrier crossing)
  - Confirmation: D_v jumps in post-barrier zone (Stage 2 slide)
  - Records WHICH pattern triggered the escape
  - Measures D_v at 4 zones: retrieval, pre-barrier, post-barrier, centroid
  - Patterns generated on GPU (no CPU→GPU transfer)
  - Rolling buffer for pre-barrier v history (no full trajectory storage)

Usage:
  julia basin_stab_LSR_v16.jl
  julia basin_stab_LSR_v16.jl --fresh

Output:
  v16_summary.csv:  per-(α,T) aggregated results
  v16_trials.csv:   per-trial escape data (t_escape, mu, phi_1mu, D_v zones)
  v16_Pesc_a{α}_T{T}.csv:  survival curves S(t)
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

const GPU_MEM_TARGET_GB = 5.0   # conservative — patterns + state + buffers
const N_DIS_MAX         = 2000
const M_PAT             = 20000
const SUMMARY_FILE      = "v16_summary.csv"
const TRIALS_FILE       = "v16_trials.csv"

# Rolling buffer size for pre-barrier D_v
const ROLL_BUF_LEN      = 200

# Post-barrier confirmation window
const CONFIRM_WINDOW    = 50    # strides to measure D_v after crossing
const DV_RATIO_THRESH   = 2.0   # D_v(post)/D_v(retrieval) must exceed this

# Centroid arrival: both φ₁ and φ_μ within δ of φ_cen
const CENTROID_DELTA_SIGMA = 2.0  # multiples of thermal σ

# Probe grid (same as v15)
const ALPHA_VALUES = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
const T_VALUES     = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80]

# Adaptive T_run and stride — fastest first (learned from v14/v15)
function get_run_params(alpha, T)
    if T >= 0.80
        return (10_000, 4)
    elseif T >= 0.50
        return (30_000, 8)
    elseif T >= 0.40
        return (100_000, 16)
    elseif T >= 0.30
        if alpha >= 0.26; return (100_000, 16)
        else;             return (300_000, 32)
        end
    elseif T >= 0.25
        if alpha >= 0.26; return (200_000, 32)
        else;             return (500_000, 64)
        end
    elseif T >= 0.20
        if alpha >= 0.26; return (300_000, 32)
        else;             return (500_000, 64)
        end
    else  # T=0.15
        if alpha >= 0.28; return (300_000, 32)
        else;             return (500_000, 64)
        end
    end
end

# Build probe list sorted by T_run (fastest first)
function build_probe_points()
    pts = Tuple{Float64,Float64,Int,Int}[]
    for α in ALPHA_VALUES, T in T_VALUES
        T_run, stride = get_run_params(α, T)
        push!(pts, (α, T, T_run, stride))
    end
    sort!(pts, by=p -> p[3])  # fastest first
    return pts
end

const PROBE_POINTS = build_probe_points()

# ══════════════════════════════════════════════════════════════════════

const F = Float16
const b_lsr      = F(2 + sqrt(2))
const PHI_C      = Float64((Float64(b_lsr)-1)/Float64(b_lsr))
const INF_ENERGY = F(1e30)

function auto_n_dis(N::Int, M::Int)
    bytes_per_dis = 2 * (N * M + N * 2 + M * 2) + ROLL_BUF_LEN * 4  # patterns + state + buffer
    return min(floor(Int, GPU_MEM_TARGET_GB * 1e9 / bytes_per_dis), N_DIS_MAX)
end

# ──────────────── LSR theory (CPU) ────────────────
function φ_eq_LSR(T)
    b = Float64(b_lsr); pc = PHI_C
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
    b = Float64(b_lsr); pc = PHI_C
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

# ──────────────── GPU kernels ────────────────
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

# ──────────────── Compute overlaps with target ────────────────
function compute_phi1!(phi_out, x, tgt, Nf)
    phi_out .= vec(sum(tgt .* x, dims=1)) ./ Nf
end

# ──────────────── Find max non-target overlap and its index ────────────────
# ov is (M, n_rep, n_dis) after energy computation, already clamped to max(0,...)
# Returns (max_val, max_idx) for μ≠1
function find_max_nontarget!(max_val, max_idx, ov)
    # Zero out pattern 1 contribution, then find max
    M = size(ov, 1)
    # ov[1,:,:] is target — set to 0 temporarily
    ov1_save = ov[1:1, :, :]
    ov[1:1, :, :] .= zero(F)
    # Max over patterns (dim 1)
    mv, mi = findmax(ov, dims=1)
    max_val .= vec(mv)
    # Extract pattern index from CartesianIndex
    max_idx .= vec([ci[1] for ci in mi])
    # Restore
    ov[1:1, :, :] .= ov1_save
    return nothing
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

    # Theory values
    φ1max = sqrt(1 - exp(-2α))
    φeq = φ_eq_LSR(T)
    φcen = φ_cen_eq(T, φ1max)
    R2 = 1 - φeq^2
    thermal_σ = sqrt(R2 / N)
    centroid_δ = CENTROID_DELTA_SIGMA * thermal_σ

    @printf("  α=%.2f, T=%.2f, N=%d, n_dis=%d, T_run=%d, stride=%d\n",
            α, T, N, n_dis, T_run, stride)
    @printf("  φ_eq=%.4f, φ_cen=%.4f, φ_c=%.4f, thermal_σ=%.4f\n",
            φeq, φcen, PHI_C, thermal_σ)

    # ── Generate patterns on GPU ──
    Random.seed!(hash((α, T, M, :v16)))
    pat_g = CUDA.randn(F, N, M, n_dis)
    # Normalize each column to √N
    norms = sqrt.(sum(pat_g .^ 2, dims=1))
    @. pat_g = sqrt(Nf) * pat_g / norms

    tgt_g = reshape(pat_g[:, 1, :], N, 1, n_dis)

    # Compute inter-pattern overlaps φ_{1μ} for each disorder sample
    # φ_{1μ} = ξ¹·ξ^μ / N
    phi_1mu_cpu = zeros(Float32, M, n_dis)
    for d in 1:n_dis
        tgt_cpu = Array(pat_g[:, 1, d])
        for mu in 1:M
            pat_cpu = Array(pat_g[:, mu, d])
            phi_1mu_cpu[mu, d] = Float32(dot(tgt_cpu, pat_cpu) / N)
        end
    end

    # Initialize at φ=1: x = ξ¹
    x_g = CUDA.zeros(F, N, n_rep, n_dis)
    for d in 1:n_dis
        x_g[:, 1, d] .= tgt_g[:, 1, d]
    end

    # Allocate MC buffers
    xp_g = similar(x_g)
    ov_g = CUDA.zeros(F, M, n_rep, n_dis)
    E_g  = CUDA.zeros(F, n_chains)
    Ep_g = CUDA.zeros(F, n_chains)
    ra_g = CUDA.zeros(F, n_chains)
    phi1_g = CUDA.zeros(F, n_chains)

    # Max non-target overlap tracking
    max_ov_val = CUDA.zeros(F, n_chains)
    max_ov_idx = CUDA.zeros(Int32, n_chains)

    compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

    # ── Per-trial state (CPU — updated at each stride) ──
    # Status: 0=retrieval, 1=candidate_escape, 2=confirmed_escape, 3=at_centroid
    status        = zeros(Int32, n_chains)
    t_escape      = fill(Int32(-1), n_chains)
    t_centroid    = fill(Int32(-1), n_chains)
    mu_escape     = zeros(Int32, n_chains)
    phi_1mu_esc   = zeros(Float32, n_chains)

    # D_v accumulators: sum of Δv², count
    Dv_sum  = zeros(Float64, n_chains, 4)  # zones 1-4
    Dv_cnt  = zeros(Int32, n_chains, 4)

    # Rolling buffer for v toward current top pattern (pre-barrier)
    roll_v   = zeros(Float32, ROLL_BUF_LEN, n_chains)
    roll_idx = zeros(Int32, n_chains)  # circular index
    roll_mu  = zeros(Int32, n_chains)  # which μ the v is computed toward

    # Confirmation counter (strides since candidate escape)
    confirm_cnt = zeros(Int32, n_chains)
    confirm_dv  = zeros(Float64, n_chains)  # Dv accumulator in confirmation window

    # Survival curve
    escape_count = zeros(Int, n_record)  # cumulative confirmed escapes

    # Previous v values for D_v computation
    v_prev = zeros(Float32, n_chains)
    v_prev_valid = falses(n_chains)

    t_start = time()
    rec_idx = 0
    prog = Progress(T_run, desc="  MC: ", showspeed=true)

    for step in 1:T_run
        mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ, ra_g)

        if step % stride == 0
            rec_idx += 1
            compute_phi1!(phi1_g, x_g, tgt_g, Nf)
            phi1_cpu = Array(phi1_g)

            # Find max non-target overlap
            find_max_nontarget!(max_ov_val, max_ov_idx, ov_g)
            max_val_cpu = Array(max_ov_val)
            max_idx_cpu = Array(max_ov_idx)

            # Process each chain
            for j in 1:n_chains
                d = (j - 1) ÷ n_rep + 1  # disorder index

                # Current top non-target pattern
                mu_top = Int(max_idx_cpu[j])
                phi_mu_top = Float32(max_val_cpu[j])  # this is ov value, not φ

                # Convert ov to φ: ov = max(0, 1-b+b*φ/N*N) → if ov > 0: φ = (ov/b_lsr + b_lsr - 1) * N / (b_lsr * ... )
                # Actually ov = max(0, 1 - b + b * φ_μ). So φ_μ = (ov + b - 1) / b if ov > 0
                # But we need the actual φ_μ from the overlap, not from ov
                # The overlap with mu_top: φ_{mu_top} = x · ξ^{mu_top} / N
                # We have phi_1mu_cpu[mu_top, d] = inter-pattern overlap
                # For v computation: v = (φ_μ - φ₁ × φ_{1μ}) / √(1 - φ_{1μ}²)
                # φ_μ at current state: need to extract from ov
                # ov[mu,:,:] = max(0, 1 - b + b × φ_μ). If ov > 0: φ_μ = (ov - 1 + b) / b
                # But ov was computed with Nf normalization: ov = max(0, 1 - b + b * (pat'x)/(N))
                # So if ov[mu] > 0: φ_μ = (ov[mu] - 1 + b_lsr) / b_lsr

                b_f64 = Float64(b_lsr)
                phi_mu = phi_mu_top > 0 ? (Float64(phi_mu_top) - 1 + b_f64) / b_f64 : 0.0

                phi_1mu_j = Float64(phi_1mu_cpu[mu_top, d])
                denom = sqrt(max(1e-10, 1 - phi_1mu_j^2))
                v_now = Float32((phi_mu - Float64(phi1_cpu[j]) * phi_1mu_j) / denom)

                if status[j] == 0  # ── RETRIEVAL ──
                    # Accumulate D_v at zone 1 (retrieval center)
                    if v_prev_valid[j] && roll_mu[j] == mu_top
                        Δv = Float64(v_now - v_prev[j])
                        Dv_sum[j, 1] += Δv^2
                        Dv_cnt[j, 1] += 1
                    end

                    # Update rolling buffer
                    ri = (roll_idx[j] % ROLL_BUF_LEN) + 1
                    roll_v[ri, j] = v_now
                    roll_idx[j] = ri
                    roll_mu[j] = Int32(mu_top)

                    v_prev[j] = v_now
                    v_prev_valid[j] = true

                    # Check for barrier crossing: φ_μ ≥ φ_c (ov > 0 means in support)
                    if phi_mu ≥ PHI_C && mu_top != 1
                        status[j] = 1  # candidate
                        t_escape[j] = Int32(step)
                        mu_escape[j] = Int32(mu_top)
                        phi_1mu_esc[j] = Float32(phi_1mu_j)
                        confirm_cnt[j] = 0
                        confirm_dv[j] = 0.0

                        # Compute D_v at zone 2 (pre-barrier) from rolling buffer
                        # Use the last entries where roll_mu == mu_top
                        prev_v = NaN32
                        for k in 1:min(ROLL_BUF_LEN, roll_idx[j])
                            idx_k = ((roll_idx[j] - k) % ROLL_BUF_LEN) + 1
                            if idx_k >= 1 && idx_k <= ROLL_BUF_LEN
                                vk = roll_v[idx_k, j]
                                if !isnan(prev_v)
                                    Δv = Float64(vk - prev_v)
                                    Dv_sum[j, 2] += Δv^2
                                    Dv_cnt[j, 2] += 1
                                end
                                prev_v = vk
                            end
                        end
                    end

                elseif status[j] == 1  # ── CANDIDATE ESCAPE ──
                    # Accumulate D_v in confirmation window (zone 3)
                    if v_prev_valid[j]
                        Δv = Float64(v_now - v_prev[j])
                        confirm_dv[j] += Δv^2
                    end
                    confirm_cnt[j] += 1
                    v_prev[j] = v_now

                    if confirm_cnt[j] >= CONFIRM_WINDOW
                        # Check if D_v jumped
                        Dv_post = confirm_dv[j] / confirm_cnt[j]
                        Dv_ret  = Dv_cnt[j,1] > 0 ? Dv_sum[j,1] / Dv_cnt[j,1] : 1e-10

                        if Dv_post > DV_RATIO_THRESH * Dv_ret
                            # Confirmed escape
                            status[j] = 2
                            Dv_sum[j, 3] = confirm_dv[j]
                            Dv_cnt[j, 3] = confirm_cnt[j]
                        else
                            # False alarm — reset to retrieval
                            status[j] = 0
                            t_escape[j] = -1
                            mu_escape[j] = 0
                            v_prev_valid[j] = false
                        end
                    end

                elseif status[j] == 2  # ── CONFIRMED ESCAPE, sliding to centroid ──
                    # Continue accumulating D_v at zone 3
                    if v_prev_valid[j]
                        Δv = Float64(v_now - v_prev[j])
                        Dv_sum[j, 3] += Δv^2
                        Dv_cnt[j, 3] += 1
                    end
                    v_prev[j] = v_now

                    # Check centroid arrival
                    φcen_j = φ_cen_eq(T, Float64(phi_1mu_esc[j]))
                    if abs(Float64(phi1_cpu[j]) - φcen_j) < centroid_δ &&
                       abs(phi_mu - φcen_j) < centroid_δ
                        status[j] = 3
                        t_centroid[j] = Int32(step)
                    end

                elseif status[j] == 3  # ── AT CENTROID ──
                    # Accumulate D_v at zone 4
                    if v_prev_valid[j]
                        Δv = Float64(v_now - v_prev[j])
                        Dv_sum[j, 4] += Δv^2
                        Dv_cnt[j, 4] += 1
                    end
                    v_prev[j] = v_now
                end
            end

            # Count confirmed escapes for survival curve
            n_esc = count(s -> s >= 2, status)
            escape_count[rec_idx] = n_esc
            ProgressMeter.update!(prog, step,
                showvalues=[(:confirmed_esc, @sprintf("%d/%d", n_esc, n_chains))])
        else
            next!(prog)
        end
    end
    CUDA.synchronize()
    t_elapsed = time() - t_start
    @printf("  Done: %.1f s (%.3f ms/step)\n", t_elapsed, 1000*t_elapsed/T_run)

    # ── Save survival curve ──
    t_arr = [(i * stride) for i in 1:n_record]
    P_arr = escape_count ./ n_chains

    pesc_file = @sprintf("v16_Pesc_a%.2f_T%.2f.csv", α, T)
    open(pesc_file, "w") do f
        write(f, "step,P_esc\n")
        for i in 1:n_record
            @printf(f, "%d,%.6f\n", t_arr[i], P_arr[i])
        end
    end
    @printf("  Saved: %s\n", pesc_file)

    # ── Save per-trial data ──
    n_confirmed = count(s -> s >= 2, status)
    @printf("  Confirmed escapes: %d/%d (%.1f%%)\n",
            n_confirmed, n_chains, 100*n_confirmed/n_chains)

    open(TRIALS_FILE, "a") do f
        for j in 1:n_chains
            status[j] < 2 && continue  # only confirmed escapes
            Dv1 = Dv_cnt[j,1] > 0 ? Dv_sum[j,1] / Dv_cnt[j,1] / (2*stride) : NaN
            Dv2 = Dv_cnt[j,2] > 1 ? Dv_sum[j,2] / Dv_cnt[j,2] / (2*stride) : NaN
            Dv3 = Dv_cnt[j,3] > 0 ? Dv_sum[j,3] / Dv_cnt[j,3] / (2*stride) : NaN
            Dv4 = Dv_cnt[j,4] > 0 ? Dv_sum[j,4] / Dv_cnt[j,4] / (2*stride) : NaN
            @printf(f, "%.2f,%.2f,%d,%d,%d,%d,%.4f,%.4e,%.4e,%.4e,%.4e\n",
                    α, T, N, t_escape[j], t_centroid[j], mu_escape[j],
                    phi_1mu_esc[j], Dv1, Dv2, Dv3, Dv4)
        end
    end

    # ── Append to summary ──
    τ_median = n_confirmed > 0 ? median(Float64.(t_escape[status .>= 2])) : NaN
    P_final = P_arr[end]
    open(SUMMARY_FILE, "a") do f
        @printf(f, "%.2f,%.2f,%d,%d,%d,%.1f,%.6f,%d\n",
                α, T, N, M, n_dis, τ_median, P_final, n_confirmed)
    end

    # Free GPU
    for arr in [pat_g, tgt_g, x_g, xp_g, ov_g, E_g, Ep_g, ra_g, phi1_g,
                max_ov_val, max_ov_idx]
        CUDA.unsafe_free!(arr)
    end
    GC.gc(); CUDA.reclaim()
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")
    fresh = "--fresh" in ARGS

    println("=" ^ 70)
    println("v16 — Kramers Escape with Confirmed Barrier Crossing")
    println("  Criterion: φ_μ ≥ φ_c + D_v jump confirmation")
    println("  Records: which pattern, D_v at 4 zones, survival S(t)")
    println("  Points: $(length(PROBE_POINTS))")
    println("=" ^ 70)

    if fresh || !isfile(SUMMARY_FILE)
        open(SUMMARY_FILE, "w") do f
            write(f, "alpha,T,N,M,n_dis,tau_median,P_esc_final,n_confirmed\n")
        end
    end
    if fresh || !isfile(TRIALS_FILE)
        open(TRIALS_FILE, "w") do f
            write(f, "alpha,T,N,t_escape,t_centroid,mu_escape,phi_1mu,Dv_retrieval,Dv_prebarrier,Dv_postbarrier,Dv_centroid\n")
        end
    end

    for (pi, (α, T, T_run, stride)) in enumerate(PROBE_POINTS)
        if !fresh && already_done(SUMMARY_FILE, α, T)
            @printf("── Point %d/%d: α=%.2f, T=%.2f — SKIP ──\n",
                    pi, length(PROBE_POINTS), α, T)
            continue
        end

        N = max(round(Int, log(M_PAT) / α), 2)
        n_dis = auto_n_dis(N, M_PAT)

        @printf("\n── Point %d/%d ──\n", pi, length(PROBE_POINTS))
        run_point!(α, T, T_run, stride, n_dis)
    end

    println("\n" * "=" ^ 70)
    println("Complete. Summary: $SUMMARY_FILE")
    println("Per-trial: $TRIALS_FILE")
    println("=" ^ 70)
end

main()

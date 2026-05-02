#=
v17 — Kramers escape with proper D_v at 4 zones
────────────────────────────────────────────────────────────────────────
Based on v16. Improvements:
  - Proper rolling buffer for pre-barrier D_v (zone 2)
  - v(t) computed toward the specific escaping pattern μ
    using its actual overlap (not the max non-target approximation)
  - D_v at all 4 zones: retrieval, pre-barrier, post-barrier, centroid

Grid: α = {0.18, 0.20, 0.22}, T = {2.0, 1.6, 1.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
Order: fix T (highest first), sweep α.

Output:
  v17_summary.csv, v17_trials.csv, v17_Pesc_a{α}_T{T}.csv
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

const GPU_MEM_TARGET_GB = 5.0
const N_DIS_MAX         = 2000
const M_PAT             = 20000
const SUMMARY_FILE      = "v17_summary.csv"
const TRIALS_FILE       = "v17_trials.csv"

# Rolling buffer for pre-barrier v history
const ROLL_BUF_LEN      = 200

# Post-barrier confirmation
const CONFIRM_WINDOW    = 50
const DV_RATIO_THRESH   = 2.0

# Centroid arrival
const CENTROID_DELTA_SIGMA = 2.0

# Grid: metastable regime (low α)
const ALPHA_VALUES = [0.18, 0.20, 0.22]
const T_VALUES     = [2.0, 1.6, 1.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

function get_run_params(alpha, T)
    if T >= 1.0
        return (10_000, 4)
    elseif T >= 0.60
        return (15_000, 4)
    elseif T >= 0.40
        return (100_000, 16)
    elseif T >= 0.30
        return (300_000, 32)
    else  # T=0.20
        return (500_000, 64)
    end
end

# Build probe list: fix T (highest first), sweep α
function build_probe_points()
    pts = Tuple{Float64,Float64,Int,Int}[]
    for T in T_VALUES  # T already ordered highest first
        for α in ALPHA_VALUES
            T_run, stride = get_run_params(α, T)
            push!(pts, (α, T, T_run, stride))
        end
    end
    return pts
end

const PROBE_POINTS = build_probe_points()

# ══════════════════════════════════════════════════════════════════════

const F = Float16
const b_lsr      = F(2 + sqrt(2))
const PHI_C      = Float64((Float64(b_lsr)-1)/Float64(b_lsr))
const INF_ENERGY = F(1e30)

function auto_n_dis(N::Int, M::Int)
    bytes_per_dis = 2 * (N * M + N * 2 + M * 2) + ROLL_BUF_LEN * 4
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

function compute_phi1!(phi_out, x, tgt, Nf)
    phi_out .= vec(sum(tgt .* x, dims=1)) ./ Nf
end

function max_nontarget_val!(max_val, ov)
    ov_notgt = @view ov[2:end, :, :]
    mv = maximum(ov_notgt, dims=1)
    max_val .= vec(mv)
    return nothing
end

function find_escape_pattern(ov, d, n_rep)
    ov_cpu = Array(ov[:, 1, d])
    ov_cpu[1] = 0
    idx = argmax(ov_cpu)
    return idx, Float64(ov_cpu[idx])
end

# Compute overlap of state x with a SPECIFIC pattern μ for disorder sample d
function compute_phi_mu!(phi_mu_out, x, pat, mu_indices, Nf, n_dis)
    # mu_indices[d] = pattern index for disorder sample d
    # phi_mu = ξ^μ · x / N
    for d in 1:n_dis
        mu = mu_indices[d]
        mu <= 0 && (phi_mu_out[d] = 0f0; continue)
        # Dot product on GPU: pat[:,mu,d] · x[:,1,d]
        phi_mu_out[d] = F(sum(pat[:, mu, d] .* x[:, 1, d]) / Nf)
    end
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

    φeq = φ_eq_LSR(T)
    φ1max = sqrt(1 - exp(-2α))
    φcen = φ_cen_eq(T, φ1max)
    R2 = 1 - φeq^2
    thermal_σ = sqrt(R2 / N)
    centroid_δ = CENTROID_DELTA_SIGMA * thermal_σ

    @printf("  α=%.2f, T=%.2f, N=%d, n_dis=%d, T_run=%d, stride=%d\n",
            α, T, N, n_dis, T_run, stride)
    @printf("  φ_eq=%.4f, φ_cen=%.4f, φ_c=%.4f\n", φeq, φcen, PHI_C)

    # Generate patterns on GPU
    Random.seed!(hash((α, T, M, :v17)))
    pat_g = CUDA.randn(F, N, M, n_dis)
    norms = sqrt.(sum(pat_g .^ 2, dims=1))
    @. pat_g = sqrt(Nf) * pat_g / norms

    tgt_g = reshape(pat_g[:, 1, :], N, 1, n_dis)

    # Inter-pattern overlaps via batched GEMM
    tgt_col = reshape(pat_g[:, 1:1, :], N, 1, n_dis)
    phi_1mu_g = CUDA.zeros(F, M, 1, n_dis)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), pat_g, tgt_col, zero(F), phi_1mu_g)
    @. phi_1mu_g = phi_1mu_g / Nf
    phi_1mu_cpu = Float32.(Array(dropdims(phi_1mu_g, dims=2)))
    CUDA.unsafe_free!(phi_1mu_g)

    # Initialize at φ=1
    x_g = CUDA.zeros(F, N, n_rep, n_dis)
    for d in 1:n_dis
        x_g[:, 1, d] .= tgt_g[:, 1, d]
    end

    xp_g = similar(x_g)
    ov_g = CUDA.zeros(F, M, n_rep, n_dis)
    E_g  = CUDA.zeros(F, n_chains)
    Ep_g = CUDA.zeros(F, n_chains)
    ra_g = CUDA.zeros(F, n_chains)
    phi1_g = CUDA.zeros(F, n_chains)
    max_ov_val = CUDA.zeros(F, n_chains)

    compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

    # Per-trial state
    status        = zeros(Int32, n_chains)
    t_escape      = fill(Int32(-1), n_chains)
    t_centroid    = fill(Int32(-1), n_chains)
    mu_escape     = zeros(Int32, n_chains)
    phi_1mu_esc   = zeros(Float32, n_chains)

    # D_v accumulators
    Dv_sum  = zeros(Float64, n_chains, 4)
    Dv_cnt  = zeros(Int32, n_chains, 4)

    # Rolling buffer: v toward the top non-target pattern
    roll_v    = zeros(Float32, ROLL_BUF_LEN, n_chains)
    roll_mu   = zeros(Int32, ROLL_BUF_LEN, n_chains)  # which μ at each buffer slot
    roll_pos  = zeros(Int32, n_chains)  # circular position
    roll_fill = zeros(Int32, n_chains)  # how many filled

    # Confirmation
    confirm_cnt = zeros(Int32, n_chains)
    confirm_dv  = zeros(Float64, n_chains)

    # Survival curve
    escape_count = zeros(Int, n_record)

    # Previous values
    phi1_prev = zeros(Float32, n_chains)
    v_prev    = zeros(Float32, n_chains)
    v_prev_valid = falses(n_chains)

    t_start = time()
    rec_idx = 0
    prog = Progress(T_run, desc="  MC: ", showspeed=true)

    for step in 1:T_run
        mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ, ra_g)

        if step % stride == 0
            rec_idx += 1
            compute_phi1!(phi1_g, x_g, tgt_g, Nf)
            max_nontarget_val!(max_ov_val, ov_g)

            phi1_cpu = Array(phi1_g)
            max_val_cpu = Array(max_ov_val)

            b_f64 = Float64(b_lsr)

            for j in 1:n_chains
                d = (j - 1) ÷ n_rep + 1
                ov_top = Float64(max_val_cpu[j])
                phi_mu_top = ov_top > 0 ? (ov_top - 1 + b_f64) / b_f64 : 0.0

                if status[j] == 0  # ── RETRIEVAL ──
                    # D_v zone 1: Δφ₁²
                    if rec_idx > 1
                        Δφ = Float64(phi1_cpu[j]) - Float64(phi1_prev[j])
                        Dv_sum[j, 1] += Δφ^2
                        Dv_cnt[j, 1] += 1
                    end
                    phi1_prev[j] = phi1_cpu[j]

                    # Rolling buffer: track v toward current top pattern
                    if ov_top > 0
                        mu_top, _ = find_escape_pattern(ov_g, d, n_rep)
                        phi_1mu_j = Float64(phi_1mu_cpu[mu_top, d])
                        denom = sqrt(max(1e-10, 1 - phi_1mu_j^2))
                        v_now = Float32((phi_mu_top - Float64(phi1_cpu[j]) * phi_1mu_j) / denom)

                        # Store in rolling buffer
                        pos = (roll_pos[j] % ROLL_BUF_LEN) + 1
                        roll_v[pos, j] = v_now
                        roll_mu[pos, j] = Int32(mu_top)
                        roll_pos[j] = pos
                        roll_fill[j] = min(roll_fill[j] + 1, ROLL_BUF_LEN)

                        # Check barrier crossing
                        if phi_mu_top ≥ PHI_C && mu_top != 1
                            status[j] = 1
                            t_escape[j] = Int32(step)
                            mu_escape[j] = Int32(mu_top)
                            phi_1mu_esc[j] = Float32(phi_1mu_j)
                            confirm_cnt[j] = 0
                            confirm_dv[j] = 0.0
                            v_prev[j] = v_now
                            v_prev_valid[j] = true

                            # D_v zone 2: from rolling buffer entries matching this μ
                            prev_v_buf = NaN32
                            n_buf = min(Int(roll_fill[j]), ROLL_BUF_LEN)
                            for k in 1:n_buf
                                idx_k = ((roll_pos[j] - k + ROLL_BUF_LEN) % ROLL_BUF_LEN) + 1
                                if roll_mu[idx_k, j] == mu_top
                                    if !isnan(prev_v_buf)
                                        Δv = Float64(roll_v[idx_k, j] - prev_v_buf)
                                        Dv_sum[j, 2] += Δv^2
                                        Dv_cnt[j, 2] += 1
                                    end
                                    prev_v_buf = roll_v[idx_k, j]
                                end
                            end
                        end
                    end

                elseif status[j] == 1  # ── CANDIDATE ESCAPE ──
                    mu_j = Int(mu_escape[j])
                    phi_1mu_j = Float64(phi_1mu_esc[j])
                    denom = sqrt(max(1e-10, 1 - phi_1mu_j^2))
                    # Get actual φ_μ for the escaping pattern (not just the max)
                    # Use ov for the specific pattern
                    ov_mu = Float64(Array(ov_g[mu_j:mu_j, 1, (j-1)÷n_rep+1])[1])
                    phi_mu_j = ov_mu > 0 ? (ov_mu - 1 + b_f64) / b_f64 : 0.0
                    v_now = Float32((phi_mu_j - Float64(phi1_cpu[j]) * phi_1mu_j) / denom)

                    if v_prev_valid[j]
                        Δv = Float64(v_now - v_prev[j])
                        confirm_dv[j] += Δv^2
                    end
                    confirm_cnt[j] += 1
                    v_prev[j] = v_now
                    v_prev_valid[j] = true

                    if confirm_cnt[j] >= CONFIRM_WINDOW
                        Dv_post = confirm_dv[j] / confirm_cnt[j]
                        Dv_ret  = Dv_cnt[j,1] > 0 ? Dv_sum[j,1] / Dv_cnt[j,1] : 1e-10
                        if Dv_post > DV_RATIO_THRESH * Dv_ret
                            status[j] = 2
                            Dv_sum[j, 3] = confirm_dv[j]
                            Dv_cnt[j, 3] = confirm_cnt[j]
                        else
                            status[j] = 0
                            t_escape[j] = -1; mu_escape[j] = 0
                            v_prev_valid[j] = false
                        end
                    end

                elseif status[j] == 2  # ── CONFIRMED, sliding ──
                    mu_j = Int(mu_escape[j])
                    phi_1mu_j = Float64(phi_1mu_esc[j])
                    denom = sqrt(max(1e-10, 1 - phi_1mu_j^2))
                    ov_mu = Float64(Array(ov_g[mu_j:mu_j, 1, (j-1)÷n_rep+1])[1])
                    phi_mu_j = ov_mu > 0 ? (ov_mu - 1 + b_f64) / b_f64 : 0.0
                    v_now = Float32((phi_mu_j - Float64(phi1_cpu[j]) * phi_1mu_j) / denom)
                    if v_prev_valid[j]
                        Δv = Float64(v_now - v_prev[j])
                        Dv_sum[j, 3] += Δv^2; Dv_cnt[j, 3] += 1
                    end
                    v_prev[j] = v_now; v_prev_valid[j] = true

                    φcen_j = φ_cen_eq(T, phi_1mu_j)
                    if abs(Float64(phi1_cpu[j]) - φcen_j) < centroid_δ &&
                       abs(phi_mu_j - φcen_j) < centroid_δ
                        status[j] = 3; t_centroid[j] = Int32(step)
                    end

                elseif status[j] == 3  # ── AT CENTROID ──
                    mu_j = Int(mu_escape[j])
                    phi_1mu_j = Float64(phi_1mu_esc[j])
                    denom = sqrt(max(1e-10, 1 - phi_1mu_j^2))
                    ov_mu = Float64(Array(ov_g[mu_j:mu_j, 1, (j-1)÷n_rep+1])[1])
                    phi_mu_j = ov_mu > 0 ? (ov_mu - 1 + b_f64) / b_f64 : 0.0
                    v_now = Float32((phi_mu_j - Float64(phi1_cpu[j]) * phi_1mu_j) / denom)
                    if v_prev_valid[j]
                        Δv = Float64(v_now - v_prev[j])
                        Dv_sum[j, 4] += Δv^2; Dv_cnt[j, 4] += 1
                    end
                    v_prev[j] = v_now; v_prev_valid[j] = true
                end
            end

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

    # Save survival curve
    t_arr = [(i * stride) for i in 1:n_record]
    P_arr = escape_count ./ n_chains

    pesc_file = @sprintf("v17_Pesc_a%.2f_T%.2f.csv", α, T)
    open(pesc_file, "w") do f
        write(f, "step,P_esc\n")
        for i in 1:n_record
            @printf(f, "%d,%.6f\n", t_arr[i], P_arr[i])
        end
    end
    @printf("  Saved: %s\n", pesc_file)

    # Save per-trial data
    n_confirmed = count(s -> s >= 2, status)
    @printf("  Confirmed escapes: %d/%d (%.1f%%)\n",
            n_confirmed, n_chains, 100*n_confirmed/n_chains)

    open(TRIALS_FILE, "a") do f
        for j in 1:n_chains
            status[j] < 2 && continue
            Dv1 = Dv_cnt[j,1] > 0 ? Dv_sum[j,1] / Dv_cnt[j,1] / (2*stride) : NaN
            Dv2 = Dv_cnt[j,2] > 1 ? Dv_sum[j,2] / Dv_cnt[j,2] / (2*stride) : NaN
            Dv3 = Dv_cnt[j,3] > 0 ? Dv_sum[j,3] / Dv_cnt[j,3] / (2*stride) : NaN
            Dv4 = Dv_cnt[j,4] > 0 ? Dv_sum[j,4] / Dv_cnt[j,4] / (2*stride) : NaN
            @printf(f, "%.2f,%.2f,%d,%d,%d,%d,%.4f,%.4e,%.4e,%.4e,%.4e\n",
                    α, T, N, t_escape[j], t_centroid[j], mu_escape[j],
                    phi_1mu_esc[j], Dv1, Dv2, Dv3, Dv4)
        end
    end

    # Summary
    τ_median = n_confirmed > 0 ? median(Float64.(t_escape[status .>= 2])) : NaN
    P_final = P_arr[end]
    open(SUMMARY_FILE, "a") do f
        @printf(f, "%.2f,%.2f,%d,%d,%d,%.1f,%.6f,%d\n",
                α, T, N, M, n_dis, τ_median, P_final, n_confirmed)
    end

    for arr in [pat_g, tgt_g, x_g, xp_g, ov_g, E_g, Ep_g, ra_g, phi1_g, max_ov_val]
        CUDA.unsafe_free!(arr)
    end
    GC.gc(); CUDA.reclaim()
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")
    fresh = "--fresh" in ARGS

    println("=" ^ 70)
    println("v17 — Kramers Escape with Proper D_v at 4 Zones")
    println("  Grid: α ∈ {0.18, 0.20, 0.22}, T ∈ {2.0..0.2}")
    println("  Order: fix T, sweep α (highest T first)")
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

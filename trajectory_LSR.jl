#=
LSR Trajectory Viewer
────────────────────────────────────────────────────────────────────────
Runs a few MC trials on CPU for a single (α, T) using v3 parameters.
Records φ at every step (equilibration + sampling) to visualize dynamics.
Outputs: trajectory_LSR_alphaXX_TXX.{csv,png}
────────────────────────────────────────────────────────────────────────
=#

using Random
using Statistics
using LinearAlgebra
using Printf
using CSV
using DataFrames
using Plots

# ──────────────── v3 parameters ────────────────

const b_lsr     = 2 + sqrt(2)   # ≈ 3.414
const phi_c     = (b_lsr - 1) / b_lsr  # ≈ 0.707
const PHI_MIN   = 0.75
const PHI_MAX   = 1.0
const N_EQ      = 2^14          # 16384 equilibration steps
const N_SAMP    = 2^12          # 4096 sampling steps
const MIN_PAT   = 20000
const MAX_PAT   = 500000
const n_alpha   = 55
const ind       = 10
const INF_ENERGY = 1e30

const alpha_vec_all = collect(0.01:0.01:0.55)
const n_patterns_vec = range(MIN_PAT^(1/ind), MAX_PAT^(1/ind), length=n_alpha) .^ ind

adaptive_ss(N::Int) = max(0.1, 2.4 / sqrt(N))

# ──────────────── N, P for a given α ────────────────

function get_NP(alpha)
    idx = findfirst(alpha_vec_all .≈ alpha)
    isnothing(idx) && error("α = $alpha not on the v3 grid (0.01:0.01:0.55)")
    P = round(Int, n_patterns_vec[idx])
    N = max(round(Int, log(P) / alpha), 2)
    return N, P, idx
end

# ──────────────── LSR energy (CPU, single state) ────────────────
# H = -(N/b) ln Σ_μ [1 - b(1 - φ^μ)]_+

function compute_energy(x::Vector{Float64}, patterns::Matrix{Float64}, N::Int)
    Nf = Float64(N)
    Nb = Nf / b_lsr
    overlaps = patterns' * x                         # [P]
    contrib = @. max(0.0, 1.0 - b_lsr + b_lsr * overlaps / Nf)
    s = sum(contrib)
    return s > 0.0 ? -Nb * log(s) : INF_ENERGY
end

# ──────────────── MC step on S^{N-1}(√N) ────────────────

function mc_step!(x::Vector{Float64}, E::Float64,
                  patterns::Matrix{Float64}, N::Int, T::Float64, σ::Float64)
    xp = x .+ σ .* randn(N)
    xp .*= sqrt(N) / norm(xp)

    Ep = compute_energy(xp, patterns, N)

    # Reject if outside all supports (Ep = +∞)
    if Ep >= INF_ENERGY
        return E, false
    end

    ΔE = Ep - E
    if ΔE <= 0 || rand() < exp(-ΔE / T)
        x .= xp
        return Ep, true
    end
    return E, false
end

# ──────────────── Run one trial, return full φ trajectory ────────────────

function run_trial(alpha::Float64, T::Float64; seed::Int=42)
    N, P, alpha_idx = get_NP(alpha)
    σ = adaptive_ss(N)

    Random.seed!(seed + alpha_idx)
    patterns = randn(N, P)
    for j in 1:P
        c = @view patterns[:, j]
        c .*= sqrt(N) / norm(c)
    end
    target = patterns[:, 1]

    phi_init = PHI_MIN + (PHI_MAX - PHI_MIN) * rand()
    x_perp = randn(N)
    x_perp .-= dot(target, x_perp) / N .* target
    x_perp ./= norm(x_perp)
    x = phi_init .* target .+ sqrt(1 - phi_init^2) .* sqrt(N) .* x_perp

    E = compute_energy(x, patterns, N)
    n_total = N_EQ + N_SAMP
    phi_traj = Vector{Float64}(undef, n_total)
    accept_count = 0

    for s in 1:n_total
        E, acc = mc_step!(x, E, patterns, N, T, σ)
        accept_count += acc
        phi_traj[s] = dot(target, x) / N
    end

    acc_rate = accept_count / n_total
    return phi_traj, N, P, phi_init, acc_rate
end

# ══════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════

alpha = 0.4
T     = 1.0
n_trials = 3

# ══════════════════════════════════════════════════════════════════════

N, P, _ = get_NP(alpha)
σ = adaptive_ss(N)

# Boltzmann φ_eq via numerical integration (finite-N, single basin)
function boltzmann_avg(N, T)
    n_pts = 5000
    phi_grid = range(phi_c + 1e-8, 1.0 - 1e-10, length=n_pts)
    dphi = phi_grid[2] - phi_grid[1]

    # u(φ) = -(1/b) ln[b(φ - φ_c)],  log_prob = log_ρ + N·u/T  (note: energy is negative)
    log_probs = [((N-2)/2) * log(1 - φ^2) + (N / (b_lsr * T)) * log(b_lsr * (φ - phi_c))
                 for φ in phi_grid]
    lp_max = maximum(log_probs)

    w = @. exp(log_probs - lp_max)
    Z = sum(w) * dphi
    return sum(collect(phi_grid) .* w) * dphi / Z
end

φ_boltz = boltzmann_avg(N, T)

# Filename tag
alpha_tag = replace(@sprintf("%.2f", alpha), "." => "")
T_tag     = replace(@sprintf("%.2f", T),     "." => "")
base_name = "trajectory_LSR_alpha$(alpha_tag)_T$(T_tag)"

println("="^70)
println("LSR Trajectory Viewer (v3 parameters)")
@printf("  α = %.2f,  T = %.2f  (N = %d, P = %d, σ = %.4f)\n", alpha, T, N, P, σ)
@printf("  φ_c = %.4f,  Boltzmann φ_eq = %.4f\n", phi_c, φ_boltz)
@printf("  Trials: %d,  Steps: %d eq + %d samp\n", n_trials, N_EQ, N_SAMP)
println("="^70)

# ──────────────── Run trials & collect trajectories ────────────────

n_total = N_EQ + N_SAMP
all_trajs = Matrix{Float64}(undef, n_total, n_trials)

for t in 1:n_trials
    seed = 1000 * round(Int, 100*alpha) + 100 * round(Int, 100*T) + t
    phi_traj, _, _, phi_init, acc_rate = run_trial(alpha, T; seed=seed)
    all_trajs[:, t] = phi_traj

    φ_samp = mean(phi_traj[N_EQ+1:end])
    @printf("  Trial %d: φ_init = %.3f, ⟨φ⟩_samp = %.4f, acc = %.1f%%\n",
            t, phi_init, φ_samp, 100 * acc_rate)
end

# ──────────────── Save CSV ────────────────

df = DataFrame(:step => 1:n_total)
for t in 1:n_trials
    df[!, Symbol("trial_$t")] = all_trajs[:, t]
end

csv_name = "$base_name.csv"
CSV.write(csv_name, df)
println("\n✓ CSV saved: $csv_name")

# ──────────────── Plot ────────────────

y_lo = floor(minimum(all_trajs) - 0.05; digits=1)
y_hi = ceil(maximum(all_trajs) + 0.05; digits=1)

p = plot(size=(900, 400), dpi=150,
         title=@sprintf("LSR trajectory:  α = %.2f,  T = %.2f  (N = %d)", alpha, T, N),
         xlabel="MC step", ylabel="Alignment φ",
         legend=:topright, ylims=(y_lo, y_hi))

vline!(p, [N_EQ], color=:gray, linestyle=:dash, linewidth=1, label="eq → samp")

hline!(p, [φ_boltz], color=:black, linewidth=2, linestyle=:dot,
       label=@sprintf("φ_eq = %.3f", φ_boltz))

hline!(p, [phi_c], color=:gray40, linewidth=1.5, linestyle=:dashdot,
       label=@sprintf("φ_c = %.3f", phi_c))

step_thin = 8
steps = 1:step_thin:n_total
for t in 1:n_trials
    φ_samp = mean(all_trajs[N_EQ+1:end, t])
    plot!(p, collect(steps), all_trajs[steps, t],
          linewidth=1, alpha=0.8,
          label=@sprintf("trial %d (⟨φ⟩=%.3f)", t, φ_samp))
end

png_name = "$base_name.png"
savefig(p, png_name)
println("✓ PNG saved: $png_name")

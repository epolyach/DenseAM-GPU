#=
LSE saddle-bulk kinetics: data boundary T_bd(α) only, no full (α, T) grid.

For each α, scan T downward starting at T_top = min(T_sp(α), 0.495).
At each T, run N_TRIALS Metropolis chains on √N · S^{N-1} for T_MC steps,
each initialized at the RB minimum (φ_1 = φ_eq(T), perp uniform on the
(N-1)-sphere of radius √N · √(1 - φ_eq²)). Compute
  Z(α, T) = φ_eq(T) - ⟨φ_1⟩.
Maintain a streak of consecutive T-cells with Z < TH_Z. When the streak
reaches WINDOW_K, record T_bd = first T in the streak and move on.

Free energy: pure saddle-bulk, no patterns.
  log Z(x) = log[ exp(N · φ_1) + exp(N · (α + g_max)) ]

Proposal:
  x_new = x + σ · η,  η ~ N(0, I_N),  then x_new ← √N · x_new / |x_new|.
  σ chosen from local curvature γ_eq(T) at the RB minimum:
    γ_eq = T · (1 + φ_eq²) / (1 - φ_eq²)²
    σ    = √(2T / γ_eq)     [N-independent; ΔF per step ~ T, acc ≈ 0.37]

Output: kinetics_boundary_LSE_N{N}_TMC{T_MC}.csv next to this script.
  columns: alpha, T_bd
Resume on re-run; pass --fresh to overwrite each file.

Usage:
  julia -t auto kinetics_boundary_LSE.jl
  julia -t auto kinetics_boundary_LSE.jl --fresh
=#

using Base.Threads
using LinearAlgebra
using Random
using Printf
using Dates

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(φ_star) + φ_star          # ≈ 0.3774

phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

# Spinodal in T at fixed α:  phi_eq(T) = α + g_max  ⇒  T_sp = (1 - φ_c²)/φ_c.
function T_spinodal(α::Float64)
    φ_c = α + g_max
    φ_c >= 1.0 && return -1.0                       # no basin: α above LSE tip
    return (1.0 - φ_c^2) / φ_c
end

# ───────── Grid & config ─────────
const N_LIST     = [100, 300, 1000, 10000]
const T_MC       = 100_000
const N_TRIALS   = 64
const ALPHA_GRID = collect(0.20:0.01:0.65)
const T_MAX      = 0.495
const dT         = 0.01
const T_MIN_SCAN = 0.005
const TH_Z       = 0.02
const WINDOW_K   = 3

const OUT_DIR = @__DIR__

# ───────── log Z(φ_1) ─────────
@inline function log_Z(retr::Float64, log_bulk::Float64)
    m = max(retr, log_bulk)
    return m + log(exp(retr - m) + exp(log_bulk - m))
end

# ───────── Metropolis chain ─────────
# Chain starts INSIDE the basin at φ_1 = φ_eq(T), with perp components uniform
# on the (N-1)-sphere of radius √N · √(1 - φ_eq²). Starting at φ_1 = 1 would
# require the chain to climb down the linear-slope branch of F before any
# escape can be observed; σ_opt sized for the basin curvature is too aggressive
# for that slope (acc ~ e^-{N σ² / 2T}) and chains stall at φ = 1, giving Z < 0
# and spurious "in-basin" cells at every T. We are measuring Kramers escape
# from the basin minimum, so init AT the minimum.
function run_chain(α::Float64, T::Float64, N::Int,
                   log_bulk::Float64, σ::Float64, rng::AbstractRNG)
    sN  = sqrt(N)
    φ_e = min(phi_eq(T), 0.999)
    x   = Vector{Float64}(undef, N)
    x[1] = sN * φ_e
    u_perp = randn(rng, N - 1)
    u_perp ./= norm(u_perp)
    scale = sN * sqrt(max(0.0, 1.0 - φ_e^2))
    @inbounds for i in 2:N
        x[i] = scale * u_perp[i - 1]
    end
    retr = sN * x[1]
    logZ = log_Z(retr, log_bulk)

    η     = Vector{Float64}(undef, N)
    x_new = Vector{Float64}(undef, N)

    @inbounds for step in 1:T_MC
        randn!(rng, η)
        @simd for i in 1:N
            x_new[i] = x[i] + σ * η[i]
        end
        nrm = norm(x_new)
        @simd for i in 1:N
            x_new[i] *= sN / nrm
        end
        retr_new = sN * x_new[1]
        logZ_new = log_Z(retr_new, log_bulk)

        log_acc = (logZ_new - logZ) / T
        if log_acc >= 0.0 || log(rand(rng)) < log_acc
            copyto!(x, x_new)
            retr = retr_new
            logZ = logZ_new
        end
    end
    return x[1] / sN
end

# ───────── Z at (α, T) over N_TRIALS chains ─────────
function compute_Z(α::Float64, T::Float64, N::Int, log_bulk::Float64)
    φ_e  = min(phi_eq(T), 0.999)
    γ_eq = T * (1 + φ_e^2) / (1 - φ_e^2)^2
    σ    = sqrt(2 * T / γ_eq)

    φs = Vector{Float64}(undef, N_TRIALS)
    @threads for t in 1:N_TRIALS
        rng = MersenneTwister(hash((:kinbd, N, α, T, t)))
        φs[t] = run_chain(α, T, N, log_bulk, σ, rng)
    end
    return phi_eq(T) - sum(φs) / N_TRIALS, σ
end

# ───────── Resume ─────────
function load_existing(csv_path::String)
    done = Set{Float64}()
    isfile(csv_path) || return done
    for line in eachline(csv_path)
        (isempty(line) || startswith(line, "#") || startswith(line, "alpha")) && continue
        f = split(line, ",")
        push!(done, parse(Float64, f[1]))
    end
    return done
end

# ───────── Boundary scan for one (N) ─────────
function scan_N(N::Int, fresh::Bool)
    csv_path = joinpath(OUT_DIR,
        @sprintf("kinetics_boundary_LSE_N%d_TMC%d.csv", N, T_MC))
    fresh && isfile(csv_path) && rm(csv_path)
    done = load_existing(csv_path)
    @info "── N=$N  T_MC=$T_MC  α remaining: $(length(ALPHA_GRID) - length(done))"

    new_file = !isfile(csv_path)
    io = open(csv_path, "a")
    if new_file
        println(io,
            "# generator=kinetics_boundary_LSE.jl  N=$N  T_MC=$T_MC  N_TRIALS=$N_TRIALS  TH_Z=$TH_Z  WINDOW_K=$WINDOW_K  dT=$dT  generated=$(Dates.now())")
        println(io, "alpha,T_bd")
        flush(io)
    end

    # Descending T-grid: 0.495, 0.485, …, 0.005
    T_desc = collect(T_MAX:-dT:T_MIN_SCAN)

    for α in ALPHA_GRID
        α in done && continue
        T_sp     = T_spinodal(α)
        if T_sp <= 0.0
            @printf(io, "%.3f,nan\n", α); flush(io)
            @info "  α=$α  above LSE tip, T_bd=nan"
            continue
        end
        T_top    = min(T_sp, T_MAX)
        log_bulk = N * α + N * g_max

        candidate_T = NaN
        streak      = 0
        for T in T_desc
            T > T_top && continue
            Z, σ = compute_Z(α, T, N, log_bulk)
            if Z < TH_Z
                streak += 1
                isnan(candidate_T) && (candidate_T = T)
                if streak >= WINDOW_K
                    break
                end
            else
                streak = 0
                candidate_T = NaN
            end
        end
        T_bd = isnan(candidate_T) ? T_MIN_SCAN : candidate_T
        @printf(io, "%.3f,%.5f\n", α, T_bd); flush(io)
        @info @sprintf("  α=%.3f  T_top=%.4f  T_bd=%.4f", α, T_top, T_bd)
    end
    close(io)
    @info "done → $csv_path"
end

function main()
    fresh = "--fresh" in ARGS
    for N in N_LIST
        scan_N(N, fresh)
    end
end

main()

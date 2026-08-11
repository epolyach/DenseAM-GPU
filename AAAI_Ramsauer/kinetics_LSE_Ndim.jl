#=
Pure-kinetics test in the LSE saddle-bulk free energy profile.

No patterns are generated. The energy depends only on φ_1 = x[1] / √N:

  log Z(x) = log[ exp(N · φ_1) + exp(N · (α + g_max)) ]
  H(x)    = -log Z(x)

Boltzmann distribution on x ∈ √N · S^{N-1}. Metropolis with Gaussian proposal
+ sphere projection.

Each trial is one independent MC chain that starts at the target,
  x = (√N, 0, …, 0)  ⇒  φ_1 = 1,
and runs for T_MC Metropolis updates. The escape fraction at each (α, T)
is read off the ensemble of final φ_1 values across N_TRIALS chains.

Parameter N is the dimension of the vector x. T_MC is the MC-step budget per
chain. No disorder; trial index just seeds the MC RNG.

Output: kinetics_LSE_Ndim_N{N}_TMC{T_MC}.csv next to this script.
  columns: alpha, T, trial, phi_final, phi_min, acc_rate
Resume on re-run; pass --fresh to overwrite.

Usage:
  julia -t auto kinetics_LSE_Ndim.jl
  julia -t auto kinetics_LSE_Ndim.jl --fresh
=#

using Base.Threads
using LinearAlgebra
using Random
using Printf
using Dates

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(φ_star) + φ_star          # ≈ 0.3774

phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

# ───────── Grid & config ─────────
const N           = 100
const T_MC        = 100_000          # MC-step budget per chain
const N_TRIALS    = 64
const ALPHA_GRID  = collect(0.20:0.01:0.65)
const T_GRID      = collect(0.005:0.010:0.495)

const OUT_DIR = @__DIR__

# ───────── log Z(φ_1) ─────────
#   log[ exp(N · φ_1) + exp(N · (α + g_max)) ]
@inline function log_Z(retr::Float64, log_bulk::Float64)
    m = max(retr, log_bulk)
    return m + log(exp(retr - m) + exp(log_bulk - m))
end

# ───────── Metropolis on x ∈ √N · S^{N-1} ─────────
function run_chain(α::Float64, T::Float64, N::Int,
                   log_bulk::Float64, rng::AbstractRNG)
    sN = sqrt(N)
    # Cold start at the target: φ_1 = 1.
    x      = zeros(N)
    x[1]   = sN
    retr   = sN * x[1]                # = N · φ_1
    logZ   = log_Z(retr, log_bulk)

    # σ from local RB curvature: ΔF per step ≈ T, acc ≈ 0.37, N-independent.
    φ_e  = min(phi_eq(T), 0.999)
    γ_eq = T * (1 + φ_e^2) / (1 - φ_e^2)^2
    σ    = sqrt(2 * T / γ_eq)
    η      = Vector{Float64}(undef, N)
    x_new  = Vector{Float64}(undef, N)

    n_acc  = 0
    φ_min  = 1.0
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
            n_acc += 1
        end

        φ_1 = x[1] / sN
        φ_1 < φ_min && (φ_min = φ_1)
    end

    return x[1] / sN, φ_min, n_acc / T_MC
end

# ───────── Resume ─────────
function load_existing(csv_path::String)
    done = Set{Tuple{Float64,Float64,Int}}()
    isfile(csv_path) || return done
    for line in eachline(csv_path)
        (isempty(line) || startswith(line, "#") || startswith(line, "alpha")) && continue
        f = split(line, ",")
        push!(done, (parse(Float64, f[1]), parse(Float64, f[2]), parse(Int, f[3])))
    end
    return done
end

function main()
    fresh    = "--fresh" in ARGS
    csv_path = joinpath(OUT_DIR,
        @sprintf("kinetics_LSE_Ndim_N%d_TMC%d.csv", N, T_MC))
    fresh && isfile(csv_path) && rm(csv_path)
    done = load_existing(csv_path)
    @info "N=$N  T_MC=$T_MC  N_TRIALS=$N_TRIALS  resume cells: $(length(done))"

    new_file = !isfile(csv_path)
    io = open(csv_path, "a")
    if new_file
        println(io,
            "# generator=kinetics_LSE_Ndim.jl  N=$N  T_MC=$T_MC  N_TRIALS=$N_TRIALS  σ=sqrt(2T/γ_eq)  generated=$(Dates.now())")
        println(io, "alpha,T,trial,phi_final,phi_min,acc_rate")
        flush(io)
    end
    io_lock = ReentrantLock()

    tasks = vec([(t, α) for t in 1:N_TRIALS, α in ALPHA_GRID])
    counter = Threads.Atomic{Int}(0)
    total   = length(tasks)

    @threads for task in tasks
        t, α = task
        if all((α, T, t) in done for T in T_GRID)
            Threads.atomic_add!(counter, 1)
            continue
        end
        log_bulk = N * α + N * g_max     # = log[ M · exp(N · g_max) ]
        for T in T_GRID
            (α, T, t) in done && continue
            rng = MersenneTwister(hash((:kinetics, N, α, T, t)))
            φ_fin, φ_min, acc = run_chain(α, T, N, log_bulk, rng)
            lock(io_lock) do
                @printf(io, "%.3f,%.5f,%d,%.6f,%.6f,%.4f\n",
                        α, T, t, φ_fin, φ_min, acc)
                flush(io)
            end
        end
        n = Threads.atomic_add!(counter, 1) + 1
        if n % max(1, total ÷ 20) == 0
            @info "N=$N  T_MC=$T_MC  $(n)/$(total)"
        end
    end
    close(io)
    @info "done → $csv_path"
end

main()

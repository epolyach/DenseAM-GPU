#=
Smoke test for the spinodal CPU MC machinery.
Tiny scale to verify code correctness + measure per-chain timing.
=#

using Random, Statistics, LinearAlgebra, Printf
using SpecialFunctions, QuadGK, Distributions
using Base.Threads

const F = Float32
const BETANET = F(1.0)

# Re-include only the helper functions, not main().
# Load the source as text, strip the final main() call, eval.
src = read(joinpath(@__DIR__, "validate_spinodal_cpu.jl"), String)
src_no_main = replace(src, r"\nmain\(\)\n*$" => "")
include_string(Main, src_no_main)

println("Smoke test ── single (α=0.30, N=200), T ∈ {0.1, 0.3, 0.6}, 1 disorder")
println("Threads available: $(nthreads())")

α = 0.30
N = 200
K_target = 10_000
phi_keep, M = pick_phi_keep(N, α, K_target)
log_C = phi_keep < 0 ? F(-1e30) : F(α*N + log_C_bulk_per(N, phi_keep, 1.0))
@printf("M=%.2e  φ_keep=%.3f  log_C=%.3f\n", Float64(M), phi_keep, Float64(log_C))

print("Generating patterns ... "); t0 = time()
patterns, target, K_used = generate_patterns(N, α, K_target, phi_keep, M, 12345)
@printf("K_used=%d   %.2f s\n", K_used, time()-t0)

for T in (0.10, 0.30, 0.60)
    for init in (:cold, :hot)
        t0 = time()
        phi = run_chain(N, T, init, patterns, target, log_C, 8000, 2000, 99 + round(Int, 100*T))
        dt = time() - t0
        @printf("  T=%.2f  init=%s  ⟨φ⟩=%+.4f   %.2f s\n", T, init, phi, dt)
    end
end

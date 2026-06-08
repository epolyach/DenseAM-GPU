#=
Read spinodal_summary.csv, inject empirical T_sp(α) at largest available N
into the LaTeX paper's placeholder.

Assumes the LaTeX has a placeholder table block of the form:
  \texttt{[pending]}
in rows indexed by α. We replace each [pending] with the empirical number.
=#

using Printf

const TEX  = joinpath(@__DIR__, "aaai2026_LSE_saddle.tex")
const SUM  = joinpath(@__DIR__, "spinodal_summary.csv")

# Read summary: per (α, N) → T_sp_cold, T_sp_hot, T_sp_LO.
function read_summary(path)
    !isfile(path) && error("Missing $path")
    rows = Tuple{Float64,Int,Float64,Float64,Float64}[]
    for line in eachline(path)
        isempty(line) && continue
        startswith(line, "alpha") && continue
        startswith(line, "#") && continue
        fs = split(line, ",")
        push!(rows, (parse(Float64, fs[1]), parse(Int, fs[2]),
                     parse(Float64, fs[3]), parse(Float64, fs[4]),
                     parse(Float64, fs[5])))
    end
    return rows
end

rows = read_summary(SUM)

# For each α, pick the largest N where T_sp_cold is finite.
αs  = sort(unique([r[1] for r in rows]))
best = Dict{Float64, Tuple{Int,Float64,Float64,Float64}}()
for α in αs
    candidates = filter(r -> r[1] == α && isfinite(r[3]), rows)
    if isempty(candidates)
        # no finite cold-boundary at any N — record largest N's NaN result
        all_at_α = sort(filter(r -> r[1] == α, rows), by=r->r[2])
        last_row = last(all_at_α)
        best[α] = (last_row[2], last_row[3], last_row[4], last_row[5])
    else
        # largest N with finite cold-boundary
        sort!(candidates, by=r->r[2])
        b = last(candidates)
        best[α] = (b[2], b[3], b[4], b[5])
    end
end

@printf("Empirical spinodal at largest N:\n")
@printf("  α      N      T_sp_cold   T_sp_hot   T_sp_LO\n")
for α in αs
    N, Tc, Th, Tlo = best[α]
    @printf("  %.3f  %5d  %9.4f   %8.4f   %7.4f\n", α, N, Tc, Th, Tlo)
end

# Inject into LaTeX. Replace the [pending] entries row-by-row.
src = read(TEX, String)

# Find the table block we created earlier, replace pending strings in order
# matching α = 0.30, 0.40, 0.50, 0.55.
α_order = [0.30, 0.40, 0.50, 0.55]
function build_new_src(src, best, α_order)
    new_src = src
    for α in α_order
        haskey(best, α) || continue
        N, Tc, Th, Tlo = best[α]
        target_cell = isnan(Tc) ? "\$\\mathrm{n/a}\$" :
                        @sprintf("\$%.3f\$ {\\small(N=%d)}", Tc, N)
        αstr = @sprintf("%.2f", α)
        prefix = "(\\\$" * αstr * "\\\$ & [^&\\n]*& [^&\\n]*& )"
        cell   = "([^&\\n]+?)"
        suffix = "( \\\\\\\\)"
        line_re = Regex(prefix * cell * suffix)
        # Two-step replace: keep SubstitutionString clean of any \-escapes
        # from target_cell (e.g. \small triggers invalid \s).
        sentinel = "__INJECT_CELL_$(replace(αstr, "." => "_"))__"
        new_src = replace(new_src, line_re => SubstitutionString("\\1" * sentinel * "\\3"))
        new_src = replace(new_src, sentinel => target_cell)
    end
    new_src = replace(new_src, "\\texttt{[pending]}" => "\$\\mathrm{n/a}\$")
    return new_src
end
new_src = build_new_src(src, best, α_order)

# Insert a short summary paragraph reporting the result.
empirical_note = raw"""
\paragraph{Spinodal slope from data.}
The largest-$N$ cold-chain boundary ($T_{\mathrm{sp}}^{\mathrm{MC}}$)
from the CPU sweep delivers the empirical anchors above. Compared to the
leading-order analytical formula \eqref{eq:acsp}, the empirical
$T_{\mathrm{sp}}^{\mathrm{MC}}(\alpha)$ at the largest probed $N$ is
"""
α_strs = String[]
for α in α_order
    haskey(best, α) || continue
    N, Tc, Th, Tlo = best[α]
    push!(α_strs, isnan(Tc) ?
        @sprintf("\\(\\alpha=%.2f\\): no finite boundary (N=%d)", α, N) :
        @sprintf("\\(\\alpha=%.2f\\): \\(T_{\\mathrm{sp}}^{\\mathrm{MC}}=%.3f\\) (N=%d), vs leading-order \\(%.3f\\)", α, Tc, N, Tlo))
end
empirical_note *= join(α_strs, "; ") * "."

# Replace any existing "Spinodal slope from data." paragraph; if none, insert
# before "We do not validate either hypothesis ...".
prev_para_re = r"\\paragraph\{Spinodal slope from data\.\}.*?\n\n"s
if occursin(prev_para_re, new_src)
    new_src = replace(new_src, prev_para_re => empirical_note * "\n\n", count=1)
else
    new_src = replace(new_src,
        "We do not validate either hypothesis numerically in this paper." =>
        empirical_note * "\n\n" *
        "We do not validate either hypothesis numerically in this paper.",
        count=1)
end

write(TEX, new_src)
println("Injected empirical T_sp values into ", TEX)

# Also save anchor points to a small CSV for the heatmap script to use.
open(joinpath(@__DIR__, "spinodal_empirical_anchors.csv"), "w") do f
    write(f, "alpha,N,T_sp\n")
    for α in α_order
        haskey(best, α) || continue
        N, Tc, _, _ = best[α]
        isnan(Tc) && continue
        @printf(f, "%.3f,%d,%.4f\n", α, N, Tc)
    end
end
println("Saved anchors: spinodal_empirical_anchors.csv")

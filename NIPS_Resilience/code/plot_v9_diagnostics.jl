#=
Diagnostic Plotter for v9 Metastability Probe
────────────────────────────────────────────────────────────────────────
Reads v9 output files and generates separate panels (86mm, png+pdf):
  1. Convergence: φ_mean vs checkpoint for each probe point
  2. Histograms: per-trial φ distribution at last checkpoint
  3. Trajectories: φ_a(t), φ_b(t) for representative trials

Output: panels_v9/ directory
────────────────────────────────────────────────────────────────────────
=#

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# ──────────────── Configuration ────────────────

out_dir = "panels_v9"
mkpath(out_dir)

const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = round(Int, FIG_W * 0.85)

const FONT_TITLE  = 9
const FONT_GUIDE  = 8
const FONT_TICK   = 7
const FONT_LEGEND = 7

default(titlefontsize=FONT_TITLE, guidefontsize=FONT_GUIDE,
        tickfontsize=FONT_TICK, legendfontsize=FONT_LEGEND)

function save_fig(p, name)
    savefig(p, joinpath(out_dir, "$name.png"))
    savefig(p, joinpath(out_dir, "$name.pdf"))
    println("  Saved: $name.png, $name.pdf")
end

# ──────────────── Read data ────────────────

df_sum = CSV.read("v9_summary.csv", DataFrame)
df_tri = CSV.read("v9_per_trial.csv", DataFrame)

# Probe points from summary
probe_points = unique(zip(df_sum.alpha, df_sum.T)) |> collect
n_probes = length(probe_points)
@printf("Loaded %d probe points\n", n_probes)

# Colors for probe points
colors = [:red, :darkorange, :purple, :green, :blue]
labels_short = [@sprintf("α=%.2f T=%.2f", α, T) for (α, T) in probe_points]

# ──────────────── 1. Convergence: φ vs checkpoint ────────────────

p = plot(xlabel="equilibration steps", ylabel="⟨φ⟩",
    title="Convergence of φ with equilibration",
    xscale=:log10, legend=:outerright,
    size=(round(Int, FIG_W * 1.4), FIG_H), dpi=FIG_DPI,
    right_margin=5Plots.mm, left_margin=3Plots.mm,
    top_margin=3Plots.mm, bottom_margin=3Plots.mm)

for (i, (α, T)) in enumerate(probe_points)
    mask = (df_sum.alpha .≈ α) .& (df_sum.T .≈ T)
    sub = df_sum[mask, :]
    plot!(p, sub.checkpoint, sub.phi_mean,
        yerror=sub.phi_std,
        marker=:circle, markersize=4, lw=1.5,
        color=colors[i], label=labels_short[i])
end
save_fig(p, "convergence_phi")

# ──────────────── 1b. Convergence: q vs checkpoint ────────────────

p = plot(xlabel="equilibration steps", ylabel="⟨q_EA⟩",
    title="Convergence of q_EA with equilibration",
    xscale=:log10, legend=:outerright,
    size=(round(Int, FIG_W * 1.4), FIG_H), dpi=FIG_DPI,
    right_margin=5Plots.mm, left_margin=3Plots.mm,
    top_margin=3Plots.mm, bottom_margin=3Plots.mm)

for (i, (α, T)) in enumerate(probe_points)
    mask = (df_sum.alpha .≈ α) .& (df_sum.T .≈ T)
    sub = df_sum[mask, :]
    plot!(p, sub.checkpoint, sub.q_mean,
        yerror=sub.q_std,
        marker=:circle, markersize=4, lw=1.5,
        color=colors[i], label=labels_short[i])
end
save_fig(p, "convergence_q")

# ──────────────── 2. Histograms: per-trial φ at last checkpoint ────────────────

last_cp = maximum(df_tri.checkpoint)

for (i, (α, T)) in enumerate(probe_points)
    mask = (df_tri.alpha .≈ α) .& (df_tri.T .≈ T) .& (df_tri.checkpoint .== last_cp)
    sub = df_tri[mask, :]
    if nrow(sub) == 0
        println("  No data for α=$α, T=$T at checkpoint $last_cp")
        continue
    end

    p = histogram(sub.phi, bins=20, normalize=:pdf,
        xlabel="φ", ylabel="density",
        title=@sprintf("φ distribution: α=%.2f, T=%.2f (cp=%dk)", α, T, last_cp÷1000),
        color=colors[i], alpha=0.7, label="",
        size=(FIG_W, FIG_H), dpi=FIG_DPI,
        right_margin=5Plots.mm, left_margin=3Plots.mm,
        top_margin=3Plots.mm, bottom_margin=3Plots.mm)

    # Add mean line
    φ_mean = mean(sub.phi)
    vline!(p, [φ_mean], color=:black, lw=2, ls=:dash, label=@sprintf("mean=%.3f", φ_mean))

    name = @sprintf("hist_phi_a%.3f_T%.3f", α, T)
    save_fig(p, name)
end

# ──────────────── 2b. Histograms: per-trial q at last checkpoint ────────────────

for (i, (α, T)) in enumerate(probe_points)
    mask = (df_tri.alpha .≈ α) .& (df_tri.T .≈ T) .& (df_tri.checkpoint .== last_cp)
    sub = df_tri[mask, :]
    nrow(sub) == 0 && continue

    p = histogram(sub.q12, bins=20, normalize=:pdf,
        xlabel="q_EA", ylabel="density",
        title=@sprintf("q_EA distribution: α=%.2f, T=%.2f", α, T),
        color=colors[i], alpha=0.7, label="",
        size=(FIG_W, FIG_H), dpi=FIG_DPI,
        right_margin=5Plots.mm, left_margin=3Plots.mm,
        top_margin=3Plots.mm, bottom_margin=3Plots.mm)

    # Add φ² line for comparison
    φ_mean = mean(sub.phi)
    vline!(p, [φ_mean^2], color=:black, lw=2, ls=:dash, label=@sprintf("φ²=%.3f", φ_mean^2))

    name = @sprintf("hist_q_a%.3f_T%.3f", α, T)
    save_fig(p, name)
end

# ──────────────── 3. Trajectories: φ(t) ────────────────

for (i, (α, T)) in enumerate(probe_points)
    traj_file = @sprintf("v9_trajectory_a%.3f_T%.3f.csv", α, T)
    if !isfile(traj_file)
        println("  No trajectory file: $traj_file")
        continue
    end

    df_traj = CSV.read(traj_file, DataFrame)
    if nrow(df_traj) == 0
        println("  Empty trajectory file: $traj_file")
        continue
    end

    trials = unique(df_traj.trial)
    n_show = min(length(trials), 4)

    p = plot(xlabel="MC step", ylabel="φ",
        title=@sprintf("φ(t) trajectory: α=%.2f, T=%.2f", α, T),
        legend=:outerright,
        size=(round(Int, FIG_W * 1.4), FIG_H), dpi=FIG_DPI,
        right_margin=5Plots.mm, left_margin=3Plots.mm,
        top_margin=3Plots.mm, bottom_margin=3Plots.mm)

    trail_colors = [:red, :blue, :green, :orange]
    for (j, tr) in enumerate(trials[1:n_show])
        mask = df_traj.trial .== tr
        sub = df_traj[mask, :]
        plot!(p, sub.step, sub.phi_a, lw=0.8, alpha=0.8,
            color=trail_colors[j], label="trial $tr")
    end

    name = @sprintf("traj_a%.3f_T%.3f", α, T)
    save_fig(p, name)
end

# ──────────────── 4. φ vs φ_max_other per trial (last checkpoint) ────────────────

p = plot(xlabel="φ", ylabel="φ_max_other",
    title="Per-trial φ vs φ_max_other",
    legend=:outerright,
    size=(round(Int, FIG_W * 1.4), FIG_H), dpi=FIG_DPI,
    right_margin=5Plots.mm, left_margin=3Plots.mm,
    top_margin=3Plots.mm, bottom_margin=3Plots.mm)

for (i, (α, T)) in enumerate(probe_points)
    mask = (df_tri.alpha .≈ α) .& (df_tri.T .≈ T) .& (df_tri.checkpoint .== last_cp)
    sub = df_tri[mask, :]
    nrow(sub) == 0 && continue
    scatter!(p, sub.phi, sub.phi_max_other,
        markersize=3, markerstrokewidth=0, alpha=0.5,
        color=colors[i], label=labels_short[i])
end
plot!(p, [0, 1], [0, 1], color=:black, lw=1.5, ls=:dash, label="φ = φ_max")
save_fig(p, "phi_vs_phimax_trials")

# ──────────────── Done ────────────────

println("\nAll panels saved to $out_dir/")

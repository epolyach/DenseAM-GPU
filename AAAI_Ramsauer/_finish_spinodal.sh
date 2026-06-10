#!/usr/bin/env bash
# Auto-pipeline: runs once spinodal_probe_cpu.csv exists.
# 1. Analyse → spinodal_summary.csv + diagnostic plot.
# 2. Inject empirical T_sp values into the LaTeX placeholder.
# 3. Add empirical (α, T_sp) markers to heatmap.
# 4. Recompile PDF.
# 5. Git add/commit/push.
set -e
cd "$(dirname "$0")"

echo "── analyse"
julia --project=@. -t auto analyse_spinodal_cpu.jl

echo "── inject empirical T_sp into LaTeX"
julia --project=@. inject_latex_empirical.jl

echo "── regenerate heatmap with empirical points"
julia plot_LSE_AAAI_heatmap.jl

echo "── recompile PDF"
pdflatex -interaction=nonstopmode aaai2026_LSE_saddle.tex > /dev/null
bibtex aaai2026_LSE_saddle > /dev/null || true
pdflatex -interaction=nonstopmode aaai2026_LSE_saddle.tex > /dev/null
pdflatex -interaction=nonstopmode aaai2026_LSE_saddle.tex > /dev/null

echo "── commit and push"
cd ..
git add AAAI_Ramsauer/aaai2026_LSE_saddle.tex \
        AAAI_Ramsauer/aaai2026_LSE_saddle.pdf \
        AAAI_Ramsauer/validate_spinodal_cpu.jl \
        AAAI_Ramsauer/analyse_spinodal_cpu.jl \
        AAAI_Ramsauer/inject_latex_empirical.jl \
        AAAI_Ramsauer/plot_LSE_AAAI_heatmap.jl \
        AAAI_Ramsauer/spinodal_probe_cpu.csv \
        AAAI_Ramsauer/spinodal_summary.csv \
        AAAI_Ramsauer/panels_paper/heatmap_LSE_AAAI_residual_3boundaries.png \
        AAAI_Ramsauer/panels_paper/heatmap_LSE_AAAI_residual_3boundaries.pdf \
        AAAI_Ramsauer/panels_paper/spinodal_cold_hot.png \
        AAAI_Ramsauer/panels_paper/spinodal_cold_hot.pdf
git commit -F /tmp/spinodal_commit_msg.txt
git push

echo "── done"

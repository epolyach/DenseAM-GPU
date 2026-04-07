#=
Schematic: basin connectivity at low vs high temperature
Left:  low T  — basins narrow, target isolated (retrieval)
Right: high T — basins widen, giant connected cluster (paramagnetic)

Patterns are shown as disks in the 2D plane; overlapping disks are connected.

Output: basin_connectivity_schematic.png
=#

using Plots, Random, Printf

Random.seed!(7)

# ── Parameters ──
n_pat = 15
# Random 2D positions (representing pattern centers projected to 2D)
px = rand(n_pat) * 10
py = rand(n_pat) * 10

target = 1  # target pattern

# Basin radii for two temperatures
r_low  = 1.05   # low T:  small disks, few overlaps
r_high = 1.80   # high T: large disks, percolation

# ── Helpers ──
dist(i, j) = sqrt((px[i] - px[j])^2 + (py[i] - py[j])^2)

function bfs_component(adj, start)
    n = size(adj, 1)
    visited = falses(n)
    queue = [start]
    visited[start] = true
    while !isempty(queue)
        node = popfirst!(queue)
        for j in 1:n
            if adj[node, j] && !visited[j]
                visited[j] = true
                push!(queue, j)
            end
        end
    end
    return visited
end

function build_adj(r)
    adj = falses(n_pat, n_pat)
    for i in 1:n_pat, j in i+1:n_pat
        if dist(i, j) < 2r   # disks overlap
            adj[i, j] = adj[j, i] = true
        end
    end
    return adj
end

# ── Build graphs ──
adj_low  = build_adj(r_low)
adj_high = build_adj(r_high)
comp_low  = bfs_component(adj_low,  target)
comp_high = bfs_component(adj_high, target)

# ── Colors ──
col_target      = RGB(0.85, 0.22, 0.22)
col_target_fill = RGBA(0.90, 0.30, 0.30, 0.15)
col_conn        = RGB(0.25, 0.55, 0.85)
col_conn_fill   = RGBA(0.30, 0.60, 0.90, 0.12)
col_iso         = RGB(0.65, 0.65, 0.65)
col_iso_fill    = RGBA(0.70, 0.70, 0.70, 0.10)
col_edge_on     = RGBA(0.20, 0.45, 0.75, 0.6)
col_edge_off    = RGBA(0.75, 0.75, 0.75, 0.4)

# ── Draw a circle (disk outline) ──
function draw_disk!(p, cx, cy, r; linecolor=:gray, fillcolor=nothing,
                    fillalpha=0.1, lw=1.5, ls=:solid)
    θ = range(0, 2π, length=120)
    x = cx .+ r .* cos.(θ)
    y = cy .+ r .* sin.(θ)
    if fillcolor !== nothing
        plot!(p, Shape(x, y), fillcolor=fillcolor, fillalpha=fillalpha,
              linecolor=linecolor, linewidth=lw, linestyle=ls, label=false)
    else
        plot!(p, x, y, lw=lw, color=linecolor, ls=ls, label=false)
    end
end

# ── Draw one panel ──
function draw_panel!(p, r, adj, comp;
                     title_str="", subtitle_str="")
    # 1) Basin disks (draw first so edges and dots are on top)
    for i in 1:n_pat
        if i == target
            fc, lc = col_target_fill, col_target
        elseif comp[i]
            fc, lc = col_conn_fill, col_conn
        else
            fc, lc = col_iso_fill, col_iso
        end
        lw = i == target ? 2.5 : 1.5
        ls = comp[i] || i == target ? :solid : :dash
        draw_disk!(p, px[i], py[i], r; linecolor=lc, fillcolor=fc,
                   fillalpha=0.15, lw=lw, ls=ls)
    end

    # 2) Connection edges
    for i in 1:n_pat, j in i+1:n_pat
        if adj[i, j]
            ec = (comp[i] && comp[j]) ? col_edge_on : col_edge_off
            lw = (comp[i] && comp[j]) ? 2.0 : 1.0
            plot!(p, [px[i], px[j]], [py[i], py[j]],
                  lw=lw, color=ec, label=false)
        end
    end

    # 3) Pattern center dots
    for i in 1:n_pat
        mc = i == target ? col_target :
             comp[i]     ? col_conn : col_iso
        ms = i == target ? 8 : 6
        msw = i == target ? 2.0 : 1.5
        scatter!(p, [px[i]], [py[i]], ms=ms, color=mc,
                 markerstrokecolor=:white, markerstrokewidth=msw,
                 label=false)
    end

    # 4) Label target
    annotate!(p, px[target] + r * 0.6, py[target] + r * 0.6,
              text("ξ¹", 13, col_target))

    # 5) Stats
    n_edges = count(adj) ÷ 2
    n_comp  = count(comp)
    annotate!(p, 5.0, -1.0,
        text(subtitle_str * @sprintf("  edges: %d   cluster: %d/%d",
             n_edges, n_comp, n_pat), 9, :gray40))

    title!(p, title_str)
end

# ── Compose ──
margin = 2.0
p1 = plot(aspect_ratio=:equal,
          xlims=(minimum(px)-r_high-margin, maximum(px)+r_high+margin),
          ylims=(minimum(py)-r_high-margin, maximum(py)+r_high+margin),
          axis=false, grid=false, framestyle=:none, legend=false,
          titlefontsize=13)
draw_panel!(p1, r_low, adj_low, comp_low;
            title_str="Low T  (retrieval)",
            subtitle_str="q_eff ≈ φ_c  ")

p2 = plot(aspect_ratio=:equal,
          xlims=(minimum(px)-r_high-margin, maximum(px)+r_high+margin),
          ylims=(minimum(py)-r_high-margin, maximum(py)+r_high+margin),
          axis=false, grid=false, framestyle=:none, legend=false,
          titlefontsize=13)
draw_panel!(p2, r_high, adj_high, comp_high;
            title_str="High T  (paramagnetic)",
            subtitle_str="q_eff < φ_c  ")

fig = plot(p1, p2, layout=(1, 2), size=(1400, 650), dpi=150,
           plot_title="Basin percolation: temperature drives connectivity",
           margin=5Plots.mm)

savefig(fig, "basin_connectivity_schematic.png")
println("✓ Saved: basin_connectivity_schematic.png")

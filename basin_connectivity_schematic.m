%% Schematic: basin connectivity at low vs high temperature
%  Left panel:  low T  — basins are narrow, target is isolated (retrieval)
%  Right panel: high T — basins widen, giant connected cluster (paramagnetic)

clear; close all;

rng(42);  % reproducible layout

%% Parameters
n_patterns = 14;                   % number of patterns
R = 1;                             % circle radius (represents S^{N-1})

% Place pattern centers at random angles on the circle
angles = sort(2*pi*rand(1, n_patterns));

% Basin half-widths (angular radius of spherical cap)
theta_low  = 0.28;    % low T:  narrow effective basins
theta_high = 0.50;    % high T: wider effective basins

target_idx = 1;       % the "target" pattern

%% Helper: check if two basins overlap
overlap = @(a1, a2, theta) ...
    min(abs(a1 - a2), 2*pi - abs(a1 - a2)) < 2*theta;

%% Helper: find connected component containing target via BFS
function comp = bfs_component(adj, start)
    n = size(adj, 1);
    comp = false(1, n);
    queue = start;
    comp(start) = true;
    while ~isempty(queue)
        node = queue(1); queue(1) = [];
        nbrs = find(adj(node, :) & ~comp);
        comp(nbrs) = true;
        queue = [queue, nbrs];
    end
end

%% Build adjacency matrices
adj_low  = false(n_patterns);
adj_high = false(n_patterns);
for i = 1:n_patterns
    for j = i+1:n_patterns
        if overlap(angles(i), angles(j), theta_low)
            adj_low(i,j) = true; adj_low(j,i) = true;
        end
        if overlap(angles(i), angles(j), theta_high)
            adj_high(i,j) = true; adj_high(j,i) = true;
        end
    end
end

% Connected components containing target
comp_low  = bfs_component(adj_low,  target_idx);
comp_high = bfs_component(adj_high, target_idx);

%% Colors
col_bg        = [0.92 0.92 0.92];  % background basin fill
col_target    = [0.85 0.25 0.25];  % target basin
col_connected = [0.30 0.65 0.90];  % connected to target
col_isolated  = [0.75 0.75 0.75];  % not connected
col_edge_on   = [0.20 0.50 0.80];  % active connection line
col_edge_off  = [0.85 0.85 0.85];  % inactive connection (shown faintly)
col_sphere    = [0.3 0.3 0.3];     % sphere outline

%% Draw one panel
function draw_panel(ax, angles, theta, adj, comp, target_idx, ...
                    col_target, col_connected, col_isolated, ...
                    col_edge_on, col_edge_off, col_sphere, title_str)
    axes(ax); hold on; axis equal off;
    n = length(angles);
    R = 1;

    % Draw sphere outline
    t = linspace(0, 2*pi, 300);
    plot(R*cos(t), R*sin(t), '-', 'Color', col_sphere, 'LineWidth', 2);

    % Draw basin arcs (as filled wedges on the circle)
    for i = 1:n
        a = angles(i);
        arc_t = linspace(a - theta, a + theta, 80);

        % Basin fill (arc + radial shading)
        r_inner = 0.82; r_outer = 1.18;
        x_fill = [r_inner*cos(arc_t), fliplr(r_outer*cos(arc_t))];
        y_fill = [r_inner*sin(arc_t), fliplr(r_outer*sin(arc_t))];

        if i == target_idx
            fc = col_target;
            fa = 0.45;
        elseif comp(i)
            fc = col_connected;
            fa = 0.35;
        else
            fc = col_isolated;
            fa = 0.25;
        end
        fill(x_fill, y_fill, fc, 'FaceAlpha', fa, 'EdgeColor', 'none');

        % Basin arc outline
        if i == target_idx
            plot(R*cos(arc_t), R*sin(arc_t), '-', 'Color', col_target, 'LineWidth', 3);
        elseif comp(i)
            plot(R*cos(arc_t), R*sin(arc_t), '-', 'Color', col_connected, 'LineWidth', 2);
        else
            plot(R*cos(arc_t), R*sin(arc_t), '-', 'Color', col_isolated, 'LineWidth', 1.5);
        end
    end

    % Draw connection edges (as chords)
    for i = 1:n
        for j = i+1:n
            if adj(i,j)
                xi = R*cos(angles(i)); yi = R*sin(angles(i));
                xj = R*cos(angles(j)); yj = R*sin(angles(j));
                if comp(i) && comp(j)
                    plot([xi xj], [yi yj], '-', 'Color', [col_edge_on 0.7], ...
                        'LineWidth', 2);
                else
                    plot([xi xj], [yi yj], '-', 'Color', [col_edge_off 0.5], ...
                        'LineWidth', 1);
                end
            end
        end
    end

    % Draw pattern centers as dots
    for i = 1:n
        xi = R*cos(angles(i)); yi = R*sin(angles(i));
        if i == target_idx
            plot(xi, yi, 'o', 'MarkerSize', 10, 'MarkerFaceColor', col_target, ...
                'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
        elseif comp(i)
            plot(xi, yi, 'o', 'MarkerSize', 7, 'MarkerFaceColor', col_connected, ...
                'MarkerEdgeColor', 'w', 'LineWidth', 1);
        else
            plot(xi, yi, 'o', 'MarkerSize', 7, 'MarkerFaceColor', col_isolated, ...
                'MarkerEdgeColor', 'w', 'LineWidth', 1);
        end
    end

    % Label target
    xt = 1.30*cos(angles(target_idx));
    yt = 1.30*sin(angles(target_idx));
    text(xt, yt, '\xi^1', 'FontSize', 14, 'FontWeight', 'bold', ...
        'Color', col_target, 'HorizontalAlignment', 'center', ...
        'Interpreter', 'tex');

    % Title and annotations
    title(title_str, 'FontSize', 14, 'FontWeight', 'bold');
    xlim([-1.5 1.5]); ylim([-1.5 1.5]);

    % Count stats
    n_edges = sum(adj(:))/2;
    n_comp  = sum(comp);
    text(0, -1.40, sprintf('q_{eff}: large    edges: %d    cluster: %d/%d', ...
        n_edges, n_comp, n), 'FontSize', 10, 'HorizontalAlignment', 'center', ...
        'Color', [0.4 0.4 0.4]);
end

%% Create figure
fig = figure('Position', [50 100 1200 550], 'Color', 'w');

ax1 = subplot(1, 2, 1);
draw_panel(ax1, angles, theta_low, adj_low, comp_low, target_idx, ...
    col_target, col_connected, col_isolated, ...
    col_edge_on, col_edge_off, col_sphere, ...
    'Low temperature (retrieval)');
% Override the q_eff annotation
children = get(ax1, 'Children');
% Replace annotation
text(0, -1.40, sprintf('q_{eff} \\approx \\phi_c    edges: %d    cluster: %d/%d', ...
    sum(adj_low(:))/2, sum(comp_low), n_patterns), ...
    'FontSize', 10, 'HorizontalAlignment', 'center', ...
    'Color', [0.4 0.4 0.4], 'Parent', ax1);

ax2 = subplot(1, 2, 2);
draw_panel(ax2, angles, theta_high, adj_high, comp_high, target_idx, ...
    col_target, col_connected, col_isolated, ...
    col_edge_on, col_edge_off, col_sphere, ...
    'High temperature (paramagnetic)');
text(0, -1.40, sprintf('q_{eff} < \\phi_c    edges: %d    cluster: %d/%d', ...
    sum(adj_high(:))/2, sum(comp_high), n_patterns), ...
    'FontSize', 10, 'HorizontalAlignment', 'center', ...
    'Color', [0.4 0.4 0.4], 'Parent', ax2);

sgtitle('Basin connectivity on S^{N-1}: percolation at finite temperature', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'tex');

print('-dpng', '-r200', 'basin_connectivity_schematic.png');
fprintf('Saved: basin_connectivity_schematic.png\n');

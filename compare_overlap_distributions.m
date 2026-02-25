%% Compare exact spherical overlap distribution vs Gaussian approximation
%  Illustrates the lighter tail of f(q) = C*(1-q^2)^((N-3)/2) relative
%  to N(0,1/N), explaining the lambda vs <K> discrepancy.

clear; close all;

N = 50;            % dimension
phi_c = 1/sqrt(2); % overlap threshold

%% 1. Densities
q = linspace(-1, 1, 2000);

% Exact spherical density: f(q) = (1-q^2)^((N-3)/2) / B(1/2,(N-1)/2)
log_f = ((N-3)/2) * log(max(1 - q.^2, 1e-300)) - betaln(0.5, (N-1)/2);
f_exact = exp(log_f);
f_exact(abs(q) >= 1) = 0;

% Gaussian approximation: N(0, 1/N)
f_gauss = sqrt(N/(2*pi)) * exp(-N * q.^2 / 2);

%% 2. Tail probabilities: Pr(q > x) for x in [0, 1)
x_tail = linspace(0, 0.95, 500);

% Exact: numerical integration
P_exact = zeros(size(x_tail));
for k = 1:length(x_tail)
    P_exact(k) = integral(@(q) exp(((N-3)/2)*log(1-q.^2) - betaln(0.5,(N-1)/2)), ...
                           x_tail(k), 1);
end

% Gaussian: erfc
P_gauss = 0.5 * erfc(x_tail * sqrt(N/2));

%% 3. Empirical: sample overlaps on S^{N-1}(sqrt(N))
n_samp = 200000;
Z = randn(N, n_samp);
norms = sqrt(sum(Z.^2, 1));
X = sqrt(N) * Z ./ norms;           % uniform on S^{N-1}(sqrt(N))
q_samp = X(1,:) / sqrt(N);          % overlap with e_1 (= first coord / sqrt(N))

%% Plot
figure('Position', [100 100 1400 500]);

% Panel 1: full densities + histogram
subplot(1,3,1); hold on;
histogram(q_samp, 150, 'Normalization', 'pdf', ...
    'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'DisplayName', ...
    sprintf('Empirical (%.0ek samples)', n_samp/1e3));
plot(q, f_exact, 'b-', 'LineWidth', 2, 'DisplayName', ...
    sprintf('Exact: (1-q^2)^{(N-3)/2}'));
plot(q, f_gauss, 'r--', 'LineWidth', 2, 'DisplayName', ...
    sprintf('Gaussian N(0,1/%d)', N));
xline(phi_c, 'k:', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('\\phi_c = 1/\\surd2'));
xlabel('q'); ylabel('f(q)');
title(sprintf('Overlap density (N = %d)', N));
legend('Location', 'northeast', 'FontSize', 8);
xlim([-0.6 0.6]);

% Panel 2: tail zoom (log scale)
subplot(1,3,2); hold on;
semilogy(x_tail, P_exact, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact sphere');
semilogy(x_tail, P_gauss, 'r--', 'LineWidth', 2, 'DisplayName', 'Gaussian');
xline(phi_c, 'k:', 'LineWidth', 1.5, 'DisplayName', '\phi_c');
xlabel('x'); ylabel('Pr(q > x)');
title(sprintf('Tail probability (N = %d)', N));
legend('Location', 'southwest', 'FontSize', 9);
ylim([1e-12 1]);

% Panel 3: tail ratio
subplot(1,3,3); hold on;
ratio = P_exact ./ max(P_gauss, 1e-300);
valid = P_gauss > 1e-15;
plot(x_tail(valid), ratio(valid), 'b-', 'LineWidth', 2, 'DisplayName', ...
    'Pr_{sphere}/Pr_{Gauss}');
yline(exp(-N/16), 'm--', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('e^{-N/16} = %.4f', exp(-N/16)));
xline(phi_c, 'k:', 'LineWidth', 1.5, 'DisplayName', '\phi_c');
xlabel('x'); ylabel('Ratio');
title(sprintf('Tail ratio: exact / Gaussian (N = %d)', N));
legend('Location', 'southwest', 'FontSize', 9);
ylim([0 1.1]);

sgtitle(sprintf('Spherical vs Gaussian overlap distribution, N = %d', N), ...
    'FontSize', 14, 'FontWeight', 'bold');

print('-dpng', '-r150', 'compare_overlap_distributions.png');
fprintf('Saved: compare_overlap_distributions.png\n');

%% Print key numbers
fprintf('\nN = %d, phi_c = %.4f\n', N, phi_c);
fprintf('Pr_exact(q > phi_c)  = %.6e\n', ...
    integral(@(q) exp(((N-3)/2)*log(1-q.^2) - betaln(0.5,(N-1)/2)), phi_c, 1));
fprintf('Pr_gauss(q > phi_c)  = %.6e\n', 0.5*erfc(phi_c*sqrt(N/2)));
fprintf('Ratio                = %.6f\n', ...
    integral(@(q) exp(((N-3)/2)*log(1-q.^2) - betaln(0.5,(N-1)/2)), phi_c, 1) / ...
    (0.5*erfc(phi_c*sqrt(N/2))));
fprintf('e^{-N/16}            = %.6f\n', exp(-N/16));

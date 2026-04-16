% demo_interpattern_overlap.m
% Demonstrates that the inter-pattern overlap phi_{1,mu} = xi^1 . xi^mu / N
% for random patterns on S^{N-1}(sqrt(N)) is approximately Gaussian(0, 1/N).

clear; close all;

N_values = [10, 30, 50, 100];
n_samples = 100000;  % number of random pattern pairs

figure('Position', [100 100 900 700]);

for k = 1:length(N_values)
    N = N_values(k);

    % Generate pairs of random patterns on S^{N-1}(sqrt(N))
    xi1 = randn(N, n_samples);
    xi1 = sqrt(N) * xi1 ./ vecnorm(xi1);  % project onto sphere of radius sqrt(N)

    xi_mu = randn(N, n_samples);
    xi_mu = sqrt(N) * xi_mu ./ vecnorm(xi_mu);

    % Inter-pattern overlap: phi_{1,mu} = xi^1 . xi^mu / N
    phi = sum(xi1 .* xi_mu, 1) / N;

    % Histogram
    subplot(2, 2, k);
    [counts, edges] = histcounts(phi, 80, 'Normalization', 'pdf');
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    bar(centers, counts, 1, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'none');
    hold on;

    % Gaussian fit: N(0, 1/N)
    x = linspace(min(phi), max(phi), 300);
    gauss = sqrt(N/(2*pi)) * exp(-N * x.^2 / 2);
    plot(x, gauss, 'r-', 'LineWidth', 2);

    % Exact distribution: f(phi) = C_N * (1 - phi^2)^{(N-3)/2}
    % C_N = Gamma(N/2) / (Gamma((N-1)/2) * sqrt(pi))
    log_CN = gammaln(N/2) - gammaln((N-1)/2) - 0.5*log(pi);
    exact = exp(log_CN + (N-3)/2 * log(max(1 - x.^2, 1e-300)));
    plot(x, exact, 'k--', 'LineWidth', 1.5);

    % Stats
    fprintf('N = %3d: mean = %+.4f, var = %.4f (theory 1/N = %.4f)\n', ...
        N, mean(phi), var(phi), 1/N);

    title(sprintf('N = %d', N), 'FontSize', 14);
    xlabel('\phi_{1\mu}');
    ylabel('PDF');
    legend('MC histogram', sprintf('Gaussian(0, 1/%d)', N), ...
           'Exact: (1-\phi^2)^{(N-3)/2}', 'Location', 'best');
    xlim([-1 1]);
    grid on;
end

sgtitle('Inter-pattern overlap \phi_{1\mu} for random patterns on S^{N-1}(\surdN)', ...
    'FontSize', 16);

saveas(gcf, 'interpattern_overlap_gaussian.png');
fprintf('\nSaved: interpattern_overlap_gaussian.png\n');

function compare_filter_coefficients(adaptive_coeffs, original_coeffs)
    % Function to compare adaptive filter coefficients with the original
    % coefficients used to generate the desired signal.
    %
    % Inputs:
    %   adaptive_coeffs - Final adaptive filter coefficients
    %   original_coeffs - Original filter coefficients used to generate the desired signal
    
    % Ensure both coefficient vectors are the same length for comparison
    if length(adaptive_coeffs) ~= length(original_coeffs)
        error('The length of adaptive coefficients and original coefficients must be the same.');
    end

    % Plot comparison
    figure;
    subplot(2, 1, 1);
    stem(1:length(original_coeffs), original_coeffs, 'filled', 'DisplayName', 'Original Coefficients');
    title('Original Filter Coefficients');
    xlabel('Coefficient Index');
    ylabel('Coefficient Value');
    grid on;
    
    subplot(2, 1, 2);
    stem(1:length(adaptive_coeffs), adaptive_coeffs, 'filled', 'DisplayName', 'Adaptive Coefficients');
    title('Adaptive Filter Coefficients (Final)');
    xlabel('Coefficient Index');
    ylabel('Coefficient Value');
    grid on;
    
    % Add legends for better visualization
    legend('show');
end
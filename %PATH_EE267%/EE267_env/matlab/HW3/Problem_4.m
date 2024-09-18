
% Parameters
N = 4;  % Size of the matrix

% Generate the Hadamard matrix for N = 4
H = hadamard(N);

% Walsh-Hadamard basis functions
% Create a 2D grid of plots to display the basis functions
figure;
colormap(gray); % Use a grayscale colormap
for u = 1:N
    for v = 1:N
        % Basis function (outer product of two Hadamard vectors)
        basis_function = H(:, u) * H(v, :);
        
        % Invert the colors to match the image (1 -> black, -1 -> white)
        basis_function = -basis_function;
        
        % Display each basis function in its corresponding grid position
        subplot(N, N, (u - 1) * N + v);
        imagesc(basis_function); % Display image
        axis off;  % Turn off the axis for cleaner view
        axis image; % Maintain aspect ratio
        
        % Add a border around the subplots (optional for clarity)
        set(gca, 'XColor', 'none', 'YColor', 'none'); % Remove tick marks
    end
end

% Add a title to the figure
sgtitle('Walsh-Hadamard Basis Functions for N = 4');

% Adjust figure settings to make it look closer to the reference figure
set(gcf, 'Position', [100, 100, 600, 600]);  % Set figure size to make it square
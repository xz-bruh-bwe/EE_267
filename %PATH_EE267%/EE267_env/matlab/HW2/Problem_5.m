% Define the matrix G
G = [
    139 144 149 153 155 155 155 155;
    144 151 153 156 159 156 156 156;
    150 155 160 163 158 156 156 156;
    159 161 162 160 160 159 159 159;
    159 160 161 162 162 155 155 155;
    161 161 161 161 160 157 157 157;
    162 162 161 163 162 157 157 157;
    162 162 161 161 163 158 158 158
];

% Add random Gaussian noise with mean 0 and standard deviation sigma = 5
sigma = 5;
noisy_G = G + sigma * randn(size(G));

% Calculate the RMSE
rmse = sqrt(mean((G(:) - noisy_G(:)).^2));

% Calculate the PSNR
peak_value = max(G(:)); % Maximum value in the original matrix
psnr_value = 20 * log10(peak_value / rmse);

% Display the results
fprintf('RMSE: %.4f\n', rmse);
fprintf('PSNR: %.4f dB\n', psnr_value);

% Define the matrix G

G = [
    139 144 149 153 155 155 155 155;
    144 151 153 156 159 156 156 156;
    150 155 160 163 158 156 156 156;
    159 161 162 160 160 159 159 159;
    159 160 161 162 162 155 155 155;
    161 161 161 161 160 157 157 157;
    162 162 161 163 162 157 157 157;
    162 162 161 161 163 158 158 158
];

% Convert matrix to uint8 type to work with entropy function
G_uint8 = uint8(G);

% Calculate the entropy
G_entropy = entropy(G_uint8);

% Display the entropy result
fprintf('Entropy of the matrix G: %.4f bits\n', G_entropy);
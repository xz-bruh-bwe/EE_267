% Define the given matrix f
%f = [3 2; 4 -2];

f = [1 0 1 0; 2 0 2 0; 0 1 0 1; -1 0 -1 0];

% Define the size of the matrix
n = size(f, 1);

% Create the transformation matrix P
P = zeros(n, n);
for j = 1:n
    for k = 1:n
        if k == 1
            P(j, k) = 1 / sqrt(n);
        else
            P(j, k) = sqrt(2 / n) * cos((2 * (j - 1) + 1) * (k - 1) * pi / (2 * n));
        end
    end
end

% Compute the DCT
F = P' * f * P;

% Display the DCT matrix
disp('DCT of the matrix f (F):');
disp(F);

% Compute the IDCT (inverse DCT)
f_reconstructed = P * F * P';

% Display the reconstructed matrix after IDCT
disp('Reconstructed matrix after IDCT:');
disp(f_reconstructed);
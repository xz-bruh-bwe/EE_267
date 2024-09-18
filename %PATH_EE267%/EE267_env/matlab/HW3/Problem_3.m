% Step 1: Generate DCT Basis Functions for 8x8 block
n = 8;  % Size of the DCT block (8x8)
P = zeros(n, n);  % Initialize the DCT transformation matrix

for j = 1:n
    for k = 1:n
        if k == 1
            P(j, k) = 1 / sqrt(n);  % k = 0 in formula
        else
            P(j, k) = sqrt(2 / n) * cos((2 * (j - 1) + 1) * (k - 1) * pi / (2 * n));  % k >= 1
        end
    end
end

% Step 2: Visualize all 64 DCT basis functions
figure;
for u = 1:n
    for v = 1:n
        % Create the DCT basis function matrix
        basis_function = P(:,u) * P(:,v)';  % Outer product of vectors
        subplot(n, n, (u-1)*n + v);  % Display in a grid
        imagesc(basis_function);
        colormap gray;
        axis off;
        title(['u=', num2str(u-1), ', v=', num2str(v-1)]);
    end
end

% Step 3: Apply DCT to an 8x8 block from the Lena image
% Load the Lena image
lena = imread("C:\Users\Baron\Desktop\EE_267_Repo\EE_267\%PATH_EE267%\EE267_env\pictures\lena.png");
lena = rgb2gray(lena);  % Convert to grayscale if necessary

% Extract an 8x8 block (for example, the top-left corner)
block = double(lena(1:8, 1:8));

% Compute the DCT of the block
DCT_block = P' * block * P;

% Display the original block and its DCT
figure;
subplot(1, 2, 1); 
imagesc(block); 
colormap gray; 
title('Original 8x8 Block'); 

subplot(1, 2, 2);
imagesc(DCT_block);
colormap gray;
title('DCT of 8x8 Block');

% Step 4: Confirm by performing IDCT and reconstructing the original block
reconstructed_block = P * DCT_block * P';

% Display the reconstructed block
figure;
imagesc(reconstructed_block);
colormap gray;
title('Reconstructed Block (After IDCT)');

% Check for accuracy (numerical differences may be due to precision)
difference = block - reconstructed_block;
disp('Difference between original and reconstructed block:');
disp(difference);
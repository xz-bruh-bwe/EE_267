% Read the grayscale image
image = imread('Image9.jpg');
if size(image, 3) == 3
    image = rgb2gray(image); % Convert to grayscale if the image is RGB
end

% Convert the image to double for processing
image_double = double(image);

% Step 1: Apply Laplacian filter
% Define Laplacian kernel
laplacian_kernel = fspecial('laplacian', 0.2);
% Apply Laplacian filter
laplacian_image = imfilter(image_double, laplacian_kernel, 'replicate');

% Step 2: Sharpen the image
% Sharpen by subtracting the Laplacian from the original image
sharpened_image_double = image_double - laplacian_image;
% Ensure values are within valid range [0, 255]
sharpened_image_double = max(min(sharpened_image_double, 255), 0);
% Convert the result back to uint8
sharpened_image = uint8(sharpened_image_double);

% Step 3: Apply power-law (gamma) transformation
gamma = 0.5; % Adjust gamma value as needed
% Normalize the image to [0, 1]
image_normalized = double(sharpened_image) / 255.0;
% Apply power-law transformation
gamma_corrected_image = image_normalized .^ gamma;
% Scale back to [0, 255]
gamma_corrected_image = uint8(gamma_corrected_image * 255);

% Display results
figure;

subplot(1, 3, 1);
imshow(image);
title('Original Image');

subplot(1, 3, 2);
imshow(sharpened_image);
title('Sharpened Image');

subplot(1, 3, 3);
imshow(gamma_corrected_image);
title('Gamma Corrected Image');
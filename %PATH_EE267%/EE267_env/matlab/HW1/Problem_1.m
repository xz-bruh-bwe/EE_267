% Read the color Lena image
color_image = imread('lena.png');  % Load the image from file

% Convert the color image to grayscale
gray_image = rgb2gray(color_image);

% Create a figure to display both images side by side
figure;

% Display the color image
subplot(1, 2, 1);  % Create a subplot with 1 row, 2 columns, and set the first plot
imshow(color_image);  % Display the color image
title('Color Image');  % Add title to the image

% Display the grayscale image
subplot(1, 2, 2);  % Set the second plot
imshow(gray_image);  % Display the grayscale image
title('Grayscale Image');  % Add title to the image
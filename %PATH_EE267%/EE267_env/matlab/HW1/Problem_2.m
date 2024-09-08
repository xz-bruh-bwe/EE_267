I=imread('Lena.png');
G=rgb2gray(I);
imshow(G)

CG=edge(G,'sobel');
imshow(CG)

CP=edge(G,'prewitt');
imshowpair(CG,CP,'montage')
figure;

sx=[-1 0 1; -1 0 1; -1 0 1];
xx=conv2(G,sx);

if isequal(CG, CP)
    disp('The matrices are the same.');
else
    disp('The matrices are different.');
end

imshow(xx)
figure;


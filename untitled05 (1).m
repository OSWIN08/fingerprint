+clc;
clear;
close all;

%% === STEP 1: Select and Read Fingerprint Image ===
[file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tif'}, 'Select a Fingerprint Image');
if isequal(file, 0)
    error('No fingerprint image selected!');
end
img = imread(fullfile(path, file));
figure, imshow(img), title('Original Fingerprint Image');

% Convert to grayscale if needed
if size(img,3)==3
    grayImg = rgb2gray(img);
else
    grayImg = img;
end

grayImg = im2double(grayImg);
grayImg = imresize(grayImg, [256 256]);

%% === STEP 2: Image Preprocessing ===
% Histogram Equalization
enhanced1 = adapthisteq(grayImg);

% Median Filtering
enhanced2 = medfilt2(enhanced1, [3 3]);

% Normalize image
enhanced2 = mat2gray(enhanced2);

figure;
subplot(1,3,1), imshow(grayImg), title('Original Grayscale');
subplot(1,3,2), imshow(enhanced1), title('After Histogram Equalization');
subplot(1,3,3), imshow(enhanced2), title('After Median Filtering');

%% === STEP 3: Feature Extraction ===
disp('Extracting image features...');

% Feature 1: Intensity
f1 = enhanced2(:);

% Feature 2: Local variance (texture)
f2 = stdfilt(enhanced2, ones(3));
f2 = f2(:);

% Feature 3: Gradient magnitude
[Gx, Gy] = imgradientxy(enhanced2);
Gmag = sqrt(Gx.^2 + Gy.^2);
f3 = Gmag(:);

% Feature 4: Gabor filter response (ridge orientation)
wavelength = 4;
orientation = 0:45:135;
gabormag = zeros([size(enhanced2), numel(orientation)]);
for i = 1:numel(orientation)
    g = gabor(wavelength, orientation(i));
    gaborMag = imgaborfilt(enhanced2, g);
    gabormag(:,:,i) = gaborMag;
end
f4 = mean(gabormag, 3);
f4 = f4(:);

% Combine features
X = [f1 f2 f3 f4];

%% === STEP 4: Generate Labels for Training (Adaptive Thresholding) ===
T = adaptthresh(enhanced2, 0.4);   % Tuned threshold
bw = imbinarize(enhanced2, T);
Y = double(bw(:));  % 1 = ridge, 0 = background

figure;
imshow(bw); title('Initial Binarized Fingerprint');

%% === STEP 5: Train SVM ===
disp('Training SVM model...');
numPixels = numel(Y);
sampleIdx = randperm(numPixels, min(20000, numPixels));
Xsample = X(sampleIdx,:);
Ysample = Y(sampleIdx);

SVMModel = fitcsvm(Xsample, Ysample, ...
    'KernelFunction','rbf', ...
    'Standardize',true);

disp('✅ SVM training complete');

%% === STEP 6: Predict Using Trained Model ===
disp('Predicting pixel classes...');
pred = predict(SVMModel, X);
bw_ml = reshape(pred, size(enhanced2));

figure;
subplot(1,2,1), imshow(bw), title('Initial Threshold');
subplot(1,2,2), imshow(bw_ml), title('ML Predicted Ridge Map');

%% === STEP 7: Post-processing (Refined) ===
bw_clean = imopen(bw_ml, strel('disk', 1));
bw_clean = imclose(bw_clean, strel('disk', 1));
bw_clean = bwmorph(bw_clean, 'thin', Inf);
bw_clean = imfill(bw_clean, 'holes');

figure;
subplot(1,2,1), imshow(bw_ml), title('Before Cleaning');
subplot(1,2,2), imshow(bw_clean), title('Enhanced Ridge Map');

%% === STEP 8: Final Comparison Display ===
figure;
imshowpair(enhanced2, bw_clean, 'montage');
title('Left: Enhanced Image | Right: Final Ridge Map');

%% === STEP 9: Save Results ===
outDir = fullfile(path, 'Results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
imwrite(bw_clean, fullfile(outDir, 'enhanced_fingerprint.png'));
disp(['✅ Enhanced fingerprint saved in: ', outDir]);

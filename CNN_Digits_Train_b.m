clear; clc; close all;

%% USER SETTINGS

dataFolder = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_B\MNIST Digit_2';

trainImageFile = fullfile(dataFolder, 'train-images-idx3-ubyte');
trainLabelFile = fullfile(dataFolder, 'train-labels-idx1-ubyte');
modelFile      = fullfile(dataFolder, 'mnist_cnn_model.mat');

rng('shuffle');

%% CHECK FILES

if ~isfile(trainImageFile)
    error('Training image file not found: %s', trainImageFile);
end

if ~isfile(trainLabelFile)
    error('Training label file not found: %s', trainLabelFile);
end

%% LOAD TRAINING DATA

fprintf('Loading training IDX files...\n');

XTrain = loadMNISTImages(trainImageFile);
YTrain = loadMNISTLabels(trainLabelFile);

if isempty(XTrain) || isempty(YTrain)
    error('Could not load the training dataset.');
end

if size(XTrain,4) ~= numel(YTrain)
    error('Number of training images and labels does not match.');
end

%% PREPROCESS

XTrain = single(XTrain) / 255;
YTrain = categorical(YTrain);

fprintf('Training samples: %d\n', numel(YTrain));

%% DEFINE CNN

layers = [
    imageInputLayer([28 28 1], 'Normalization', 'none', 'Name', 'input')

    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(10, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% TRAIN OPTIONS

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% TRAIN NETWORK

fprintf('Training CNN...\n');
net = trainNetwork(XTrain, YTrain, layers, options);

%% SAVE MODEL

save(modelFile, 'net');

fprintf('\nModel saved to:\n%s\n', modelFile);

%% ===== LOCAL FUNCTIONS =====

function images = loadMNISTImages(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open image file: %s', filename);
    end

    magicNum = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magicNum ~= 2051
        fclose(fid);
        error('Invalid MNIST image file: %s', filename);
    end

    numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
    numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
    numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');

    rawData = fread(fid, inf, 'uint8=>uint8');
    fclose(fid);

    expectedNumPixels = numImages * numRows * numCols;
    if numel(rawData) ~= expectedNumPixels
        error('Image file size does not match header information.');
    end

    images = reshape(rawData, numCols, numRows, 1, numImages);
    images = permute(images, [2 1 3 4]);  
end

function labels = loadMNISTLabels(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open label file: %s', filename);
    end

    magicNum = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magicNum ~= 2049
        fclose(fid);
        error('Invalid MNIST label file: %s', filename);
    end

    numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
    labels = fread(fid, inf, 'uint8=>uint8');
    fclose(fid);

    if numel(labels) ~= numLabels
        error('Label file size does not match header information.');
    end
end
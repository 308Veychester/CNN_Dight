clear; clc; close all;

%% USER SETTINGS

dataFolder = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_B\MNIST Digit_1';

trainFile = fullfile(dataFolder, 'mnist_train.csv');
modelFile = fullfile(dataFolder, 'mnist_cnn_model.mat');

rng('shuffle');

%% CHECK FILE

if ~isfile(trainFile)
    error('Training file not found: %s', trainFile);
end

%% LOAD TRAINING DATA

fprintf('Loading training CSV file...\n');

trainData = readmatrix(trainFile);

if isempty(trainData)
    error('Could not read the training CSV file.');
end

YTrain = trainData(:,1);
XTrain = trainData(:,2:end);

if size(XTrain,2) ~= 784
    error('Expected 784 pixel columns after the label column.');
end

%% PREPROCESS

XTrain = single(XTrain) / 255;
YTrain = categorical(YTrain);

XTrain = reshape(XTrain', 28, 28, 1, []);

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
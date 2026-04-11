clear; clc; close all;

%% USER SETTINGS

dataFolder = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_B\MNIST Digit_1';

testFile  = fullfile(dataFolder, 'mnist_test.csv');
modelFile = fullfile(dataFolder, 'mnist_cnn_model.mat');

rng('shuffle');

%% CHECK FILES

if ~isfile(testFile)
    error('Test file not found: %s', testFile);
end

if ~isfile(modelFile)
    error('Saved model not found: %s\nRun train_mnist_cnn.m first.', modelFile);
end

%% LOAD MODEL

fprintf('Loading trained model...\n');
S = load(modelFile);

if ~isfield(S, 'net')
    error('The model file does not contain variable "net".');
end

net = S.net;

%% LOAD TEST DATA

fprintf('Loading test CSV file...\n');

testData = readmatrix(testFile);

if isempty(testData)
    error('Could not read the test CSV file.');
end

YTest = testData(:,1);
XTest = testData(:,2:end);

if size(XTest,2) ~= 784
    error('Expected 784 pixel columns after the label column.');
end

%% PREPROCESS

XTest = single(XTest) / 255;
YTest = categorical(YTest);

XTest = reshape(XTest', 28, 28, 1, []);

fprintf('Test samples: %d\n', numel(YTest));

%% FULL TEST ACCURACY

YPredAll = classify(net, XTest);
fullAccuracy = mean(YPredAll == YTest);

fprintf('\n===== FULL MNIST TEST SET =====\n');
fprintf('Full test accuracy: %.2f %%\n', 100*fullAccuracy);

%% CONFUSION MATRIX

figure('Name','Confusion Matrix','NumberTitle','off');
confusionchart(YTest, YPredAll);
title(sprintf('Confusion Matrix | Full Test Accuracy = %.2f%%', 100*fullAccuracy));

%% RANDOMLY PICK 25 TEST SAMPLES

numShow = 25;
numTest = numel(YTest);

if numTest < numShow
    error('Test set has fewer than 25 samples.');
end

randIdx = randperm(numTest, numShow);

XTest25 = XTest(:,:,:,randIdx);
YTest25 = YTest(randIdx);

%% PREDICT 25 SAMPLES

YPred25 = classify(net, XTest25);
accuracy25 = mean(YPred25 == YTest25);

fprintf('\n===== RANDOM 25-SAMPLE TEST RESULTS =====\n');
fprintf('Correct: %d / %d\n', sum(YPred25 == YTest25), numShow);
fprintf('Accuracy: %.2f %%\n', 100*accuracy25);

%% DISPLAY 25 RESULTS

figure('Name','25 Random MNIST Test Samples','NumberTitle','off');

for i = 1:numShow
    subplot(5,5,i);

    img = XTest25(:,:,1,i);
    imshow(img, []);

    trueLabel = string(YTest25(i));
    predLabel = string(YPred25(i));

    if YPred25(i) == YTest25(i)
        resultStr = 'Correct';
    else
        resultStr = 'Wrong';
    end

    title({
        ['True: ' char(trueLabel)]
        ['Pred: ' char(predLabel)]
        resultStr
        }, 'FontSize', 9);
end

sgtitle(sprintf('Random 25 MNIST Test Samples | Accuracy = %.2f%%', 100*accuracy25));

%% SHOW ALL WRONG PREDICTIONS IN MULTIPLE FIGURES

wrongIdx = find(YPredAll ~= YTest);

if isempty(wrongIdx)
    fprintf('\nNo wrong predictions in the test set.\n');
else
    numPerFigure = 25;   
    numWrong = numel(wrongIdx);
    numFigures = ceil(numWrong / numPerFigure);

    fprintf('\nTotal wrong predictions: %d\n', numWrong);
    fprintf('Number of figures needed: %d\n', numFigures);

    for figNum = 1:numFigures
        figure('Name', sprintf('Wrong Predictions %d of %d', figNum, numFigures), ...
               'NumberTitle', 'off');

        startIdx = (figNum - 1) * numPerFigure + 1;
        endIdx = min(figNum * numPerFigure, numWrong);

        for k = startIdx:endIdx
            subplotIdx = k - startIdx + 1;
            subplot(5,5,subplotIdx);

            img = XTest(:,:,1,wrongIdx(k));
            imshow(img, []);

            trueLabel = string(YTest(wrongIdx(k)));
            predLabel = string(YPredAll(wrongIdx(k)));

            title({
                ['True: ' char(trueLabel)]
                ['Pred: ' char(predLabel)]
                'Wrong'
                }, 'FontSize', 8);
        end

        sgtitle(sprintf('Wrong Predictions | Figure %d of %d', figNum, numFigures));
    end
end
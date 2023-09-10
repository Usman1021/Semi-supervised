%% How semi-supervised learning works
%%
% This code can be runed directly without changing anything by providing
% input path of your own directoty in MATLAB2023
% Give a path to your directory for training data
rootFolder = fullfile('D:OMI\trainset');
categories  = {'live','spoof'};
labeldataset = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
trainingLabels1 = labeldataset.Labels;
tbl = countEachLabel(labeldataset);
%%
% % Give a path to your directory for valiadation data, 
rootFolder = fullfile('D:OMI\tesetset');
categories  = {'live','spoof'};
imdsValidation  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%%
% input for fine-tunning using Densenet201 acrhitecture
net = densenet201;
net.Layers(1)
inputSize = net.Layers(1).InputSize;
% Data Augmentation 
 imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

 augimdsTrain1 = augmentedImageDatastore(inputSize(1:2),labeldataset, ...
     'DataAugmentation',imageAugmenter);

lgraph = layerGraph(net);

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(height(tbl), ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
layers = lgraph.Layers;
connections = lgraph.Connections;
% freeze the weights of the first ten layers
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
% Resize the images
imdsValidation.ReadFcn = @(filename)readAndPreprocessImage(filename);
% chosse the training options
options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
      'MiniBatchSize',50, ...
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',500, ...
     'ValidationData',{augimdsTrain1,imdsValidation}, ...
     'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,1));
% Fine-tune the densenet architecture
net1 = trainNetwork(augimdsTrain1,lgraph,options); 
%%
% performance evaluation for supervised learning using Equal error rate and
% setting theshold for HTER on final testing set
validationlabels = imdsValidation.Labels;
imdsValidation.ReadFcn = @(filename)readAndPreprocessImage(filename);
[~, devlpscores1] = classify(net1,imdsValidation);
% Converting labels into numerical form
 numericLabels1 = grp2idx(validationlabels);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 [~,~,Info]=vl_roc(numericLabels1,devlpscores1(:,1));
 EER = Info.eer*100
 threashold1 = Info.eerThreshold;
 %%
% Evaluate the final model on a test set (you should have a separate test dataset)
rootFolder = fullfile('D:\CASIA Dataset\testing set\testset');
categories  = {'live','spoof'};
imdsTest  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
 imdsTestlabels = imdsTest.Labels;
imdsTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
 % Evaluation for testing set interms of HTER using EER threshold
 [~,  testscores1] = classify(net1,imdsTest);
 numericLabels = grp2idx(imdsTestlabels);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores1(numericLabels==1);
 attack_scores2 =  testscores1(numericLabels==-1);
 FAR = sum(attack_scores2>threashold1) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold1) / numel(real_scores1)*100;
 HTER1 = (FAR+FRR)/2
[~,~,~,AUC1] = perfcurve(numericLabels, testscores1(:,1),1);
AUC1
%%
% Self-supervised learning stage after fine-tuned the model on labeled data
% Repeat this for multiple iterations
numIterations = 2;
confidenceThreshold = 0.9; % Adjust the confidence threshold as needed

 % Regularization hyperparameter
% Self-training loop
for epoch = 1:numIterations
   rootFolder = fullfile('D:\unlabled data');
categories  = {'50','80'};
imdsUnlabeled  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
   imdsUnlabeled.ReadFcn = @(filename)readAndPreprocessImage(filename);
    % Make predictions on unlabeled data using the current model
    [~,  predictions]  = classify(net1, imdsUnlabeled);
  % Filter high-confidence predictions
    confidence = max(predictions, [], 2);
    highConfidenceIndices = confidence >= confidenceThreshold;

    % Add high-confidence predictions to the labeled dataset
    highConfidenceImages = imdsUnlabeled.Files(highConfidenceIndices);
    % highConfidenceLabels = predictions(highConfidenceIndices);

% Initialize an empty cell array to store labels
assignedLabels = cell(size(predictions, 1), 1);

% Loop through each prediction
for i = 1:size(predictions, 1)
    % Check if the highest probability exceeds the confidence threshold
    if max(predictions(i, :)) >= confidenceThreshold
        % Assign the label based on the highest probability
        [~, idx] = max(predictions(i, :)); % Find the index of the highest probability
        if idx == 1
            assignedLabels{i} = 'live'; % Assign "real" label
        else
            assignedLabels{i} = 'spoof'; % Assign "attack" label
        end
    end
end
% Remove empty cells (undecided predictions)
assignedLabels = assignedLabels(~cellfun('isempty', assignedLabels));
% Convert the cell array of labels to a categorical array
assignedLabels = categorical(assignedLabels);
concatenatedData = [labeldataset.Files; highConfidenceImages];
labels = categorical([labeldataset.Labels; assignedLabels]);  
concatenatedDatastore = imageDatastore(concatenatedData, 'Labels', labels);
concatenatedDatastore.ReadFcn = @(filename)readAndPreprocessImage(filename);

lambda = 1.5;
options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
      'MiniBatchSize',50, ...
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',500, ...
     'ValidationData',{concatenatedDatastore,imdsValidation}, ...
     'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
     'L2Regularization', lambda, ... % Add the regularization parameter lambda
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,1));
net1 = trainNetwork(concatenatedDatastore, lgraph, options);
end
%%
% Evaluate the semi-supervised learning performance without RNN (lstm)
validationlabels = imdsValidation.Labels;
imdsValidation.ReadFcn = @(filename)readAndPreprocessImage(filename);
[~, devlpscores2] = classify(net1,imdsValidation);
% Converting labels into numerical form
 numericLabels1 = grp2idx(validationlabels);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 [~,~,Info]=vl_roc(numericLabels1,devlpscores2(:,1));
 EER = Info.eer*100
 threashold2 = Info.eerThreshold;
 %%
% Evaluate the final model on a test set (you should have a separate test dataset)
rootFolder = fullfile('D:\CASIA Dataset\testing set\testset');
categories  = {'real1','attack1'};
imdsTest  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
 imdsTestlabels = imdsTest.Labels;
imdsTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
 % Evaluation for testing set for HTER 
 [~,  testscores2] = classify(net1,imdsTest);
 numericLabels = grp2idx(imdsTestlabels);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores2(numericLabels==1);
 attack_scores2 =  testscores2(numericLabels==-1);
 FAR = sum(attack_scores2>threashold2) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold2) / numel(real_scores1)*100;
 HTER2 = (FAR+FRR)/2
[~,~,~,AUC2] = perfcurve(numericLabels, testscores2(:,1),1);
AUC2
 
%% Features extraction based on the last average pooling layer of Semi-supervised learning stage for the training set 
featureLayer = 'avg_pool' ;
labeldataset.ReadFcn = @(filename)readAndPreprocessImage(filename);
trainingFeatures1 = activations(net1, labeldataset, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer for the development set 
imdsValidation.ReadFcn = @(filename)readAndPreprocessImage(filename);
developmentFeatures1 = activations(net1, imdsValidation, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer for the testing set 
% Extract testing set features using the CNN
imdsTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures1 = activations(net1, imdsTest, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%%
% LSTM training
rng(13);
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels1';

train1 = {};
train1{end+1} = developmentFeatures1;
% 
train2 = {};
train2{end+1} = validationlabels';


numFeatures = 1920;
 numHiddenUnits =100;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
         lstmLayer(numHiddenUnits,'OutputMode','sequence')
     fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
    'MaxEpochs',100, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
       'InitialLearnRate',0.0001, ...
      'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

LSTM = trainNetwork(trainf',trainlabl,layers,options);

[~, devlp_scores3] = classify(LSTM, developmentFeatures1);
% Converting labels into numerical form
 numericLabels1 = grp2idx(validationlabels);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores3 =devlp_scores3';

 % Evaluation for testing set for EER
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores3(:,1));
 EER = Info.eer*100
 threashold3 = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [~,  testscores3] = classify(LSTM, testingFeatures1);
 testscores3 =testscores3';
 numericLabels = grp2idx(imdsTestlabels);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores3(numericLabels==1);
 attack_scores2 =  testscores3(numericLabels==-1);
 FAR = sum(attack_scores2>threashold3) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold3) / numel(real_scores1)*100;
 HTER3 = (FAR+FRR)/2
[~,~,~,AUC3] = perfcurve(numericLabels, testscores3 (:,1),1);
AUC3

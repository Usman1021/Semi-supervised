
% Load the pre-trained DenseNet-201 model Using MATLAB 2023
net = densenet201;
%%
% Create imageDatastore for labeled and unlabeled data
rootFolder = fullfile('D:\ICASSP\four datasets\ICM to O\trainset');
categories  = {'real1','attack1'};
imdsLabeled = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
tbl = countEachLabel(imdsLabeled);
%%
% Create imageDatastore unlabeled data
rootFolder = fullfile('D:\ICASSP\four datasets\ICM to O\unlabled data');
categories  = {'50','80'};
imdsUnlabeled  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%%
% Create imageDatastore validation data
rootFolder = fullfile('D:\ICASSP\four datasets\ICM to O\testseet');
categories  = {'real1','attack1'};
imdsValidation  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

%%
% Densenet Model Initialization
net.Layers(1)
inputSize = net.Layers(1).InputSize;

 imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
 augimdsTrain1 = augmentedImageDatastore(inputSize(1:2),imdsLabeled, ...
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
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%%
% Set training options and Semi-supervised learning
miniBatchSize = 32;
numEpochs = 5;
initialLearningRate = 0.001;
lambda = 0.5; % Adjust the weight for labeled data
imdsValidation.ReadFcn = @(filename)readAndPreprocessImage(filename);
imdsUnlabeled.ReadFcn = @(filename)readAndPreprocessImage(filename);
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
      'ExecutionEnvironment','gpu', ... 
    'MaxEpochs', numEpochs, ...
    'InitialLearnRate', initialLearningRate, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...
    'ExecutionEnvironment', 'auto', ...
    'GradientThresholdMethod', 'absolute', ...
    'GradientThreshold', 1, ... % Adjust the gradient threshold as needed
    'OutputFcn', @(info)customOutputFunction(info, net, imdsUnlabeled, lambda));

% Fine-tune the model with mixed labeled and pseudo-labeled data
net = trainNetwork(augimdsTrain1, lgraph, options);
%%
% Evaluate the model on a validation set to compute EER
validationlabels = imdsValidation.Labels;
imdsValidation.ReadFcn = @(filename)readAndPreprocessImage(filename);
[~, devlpscores1] = classify(net,imdsValidation);
% Converting labels into numerical form
 numericLabels1 = grp2idx(validationlabels);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 [~,~,Info]=vl_roc(numericLabels1,devlpscores1(:,1));
 EER = Info.eer*100
 threashold1 = Info.eerThreshold;
% Evaluate the final model on a test set (you should have a separate test dataset)
rootFolder = fullfile('D:\ICASSP\four datasets\Oulu testing set\testset');
categories  = {'real1','attack1'};
imdsTest  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
 imdsTestlabels = imdsTest.Labels;
imdsTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
 % Evaluation for testing set for HTER 
 [~,  testscores1] = classify(net,imdsTest);
 numericLabels = grp2idx(imdsTestlabels);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores1(numericLabels==1);
 attack_scores2 =  testscores1(numericLabels==-1);
 FAR = sum(attack_scores2>threashold1) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold1) / numel(real_scores1)*100;
 HTER1 = (FAR+FRR)/2
%%
%Semi-supervised with RNN
% Features extraction based on the last average pooling layer of ResNet for the training set 
featureLayer = 'avg_pool' ;
imdsLabeled.ReadFcn = @(filename)readAndPreprocessImage(filename);
trainingFeatures1 = activations(net, imdsLabeled, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the development set 
imdsValidation.ReadFcn = @(filename)readAndPreprocessImage(filename);
developmentFeatures1 = activations(net, imdsValidation, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Features extraction based on the last average pooling layer of ResNet for the testing set 
% Extract testing set features using the CNN
imdsTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures1 = activations(net, imdsTest, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%%
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
    'GradientThreshold',1, ...
      'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

RNN = trainNetwork(trainf',trainlabl,layers,options);

[~, devlp_scores3] = classify(RNN, developmentFeatures1);
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
 [~,  testscores3] = classify(RNN, testingFeatures1);
 testscores3 =testscores3';
 numericLabels = grp2idx(imdsTestlabels);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores3(numericLabels==1);
 attack_scores2 =  testscores3(numericLabels==-1);
 FAR = sum(attack_scores2>threashold3) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold3) / numel(real_scores1)*100;
 HTER2 = (FAR+FRR)/2

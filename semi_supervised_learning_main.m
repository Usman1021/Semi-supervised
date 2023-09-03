
% Define the path to your dataset folder (contains subfolders for each class)
%% Input for Training set 
rootFolder = fullfile('D:\four datasets\OCM to I\trainset');
categories  = {'real','attack'};
imdsLabeled = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
trainingLabels = imdsLabeled.Labels;
tbl = countEachLabel(imdsLabeled);

 %% Input for development set 
rootFolder = fullfile('D:\four datasets\OCM to I\testset');
categories  = {'real','attack'};
DevelopmentSet  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
 
%% Input for UNLABELED DATA
rootFolder = fullfile('D:\four datasets\OCM to I\unlabled data');
categories  = {'50','80'};
imdsUnlabeled  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%%
% Evaluate the final model on a test set (you should have a separate test dataset)
rootFolder = fullfile('D:\four datasets\Replay Attack dataset\test\test');
categories  = {'real','attack'};
TestingSet  = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
 TestLables = TestingSet.Labels;
 
%% Model Input
net = densenet201;
net.Layers(1)
inputSize = net.Layers(1).InputSize;
% Data augmentation
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
% Weights freezing
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
% Image resizing
DevelopmentSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
      'MiniBatchSize',32, ...
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',5, ...
     'ValidationData',{augimdsTrain1,DevelopmentSet}, ...
     'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
       'Verbose', false);
% Model learning
CNN1 = trainNetwork(augimdsTrain1,lgraph,options); 
validationlabels = DevelopmentSet.Labels;
% Image resizing 
DevelopmentSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
[~, devlpscores1] = classify(CNN1,DevelopmentSet);
% Converting labels into numerical form
 numericLabels1 = grp2idx(validationlabels);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 [~,~,Info]=vl_roc(numericLabels1,devlpscores1(:,1));
 EER = Info.eer*100
 threashold1 = Info.eerThreshold;
 %%
 % Image resizing
TestingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
 % Evaluation for testing set for HTER 
 [~,  testscores1] = classify(CNN1,TestingSet);
 numericLabels = grp2idx(TestLables);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores1(numericLabels==1);
 attack_scores2 =  testscores1(numericLabels==-1);
 FAR = sum(attack_scores2>threashold1) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold1) / numel(real_scores1)*100;
 HTER1 = (FAR+FRR)/2
[x1,y1,~,AUC1] = perfcurve(numericLabels, testscores1(:,1),1);
AUC1
plot(x1,y1,'-g','LineWidth',1.8,'MarkerSize',1.8)
grid on
hold on
%%
% Sem-Supervised learning stage
% Self-training loop (repeat this for multiple iterations)
numIterations = 3;
confidenceThreshold = 0.9; % Adjust the confidence threshold as needed

for iter = 1:numIterations
imdsUnlabeled.ReadFcn = @(filename)readAndPreprocessImage(filename);
    % Use the current model to predict labels for unlabeled data
    predictedLabels = classify(CNN1, imdsUnlabeled);
    predictedLabelsNumeric = double(predictedLabels);
    confidenceScores = max(softmax(predictedLabelsNumeric), [], 2);
    confidentIndices = confidenceScores > confidenceThreshold;
    confidentSamples = imdsUnlabeled.Files(confidentIndices);
    confidentLabels = predictedLabels(confidentIndices);
    % Add confident samples to the labeled data
    imdsLabeled.Files = [imdsLabeled.Files; confidentSamples];
    imdsLabeled.Labels = [trainingLabels ; confidentLabels];
    % Remove confident samples from the unlabeled data
    imdsUnlabeled.Files(confidentIndices) = [];
 % augimdsTrain1.ReadFcn = @(filename)readAndPreprocessImage(filename);
    % Train the model with the updated labeled data
  CNN2 = trainNetwork(augimdsTrain1, lgraph, options);
end

validationlabels = DevelopmentSet.Labels;
DevelopmentSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
[~, devlpscores2] = classify(CNN2,DevelopmentSet);
% Converting labels into numerical form
 numericLabels1 = grp2idx(validationlabels);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 [~,~,Info]=vl_roc(numericLabels1,devlpscores2(:,1));
 EER = Info.eer*100
 threashold2 = Info.eerThreshold;
 %%
% imdsTestlabels = imdsTest.Labels;
TestingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
 % Evaluation for testing set for HTER 
 [~,  testscores2] = classify(CNN2,TestingSet);
 numericLabels = grp2idx(TestLables);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores2(numericLabels==1);
 attack_scores2 =  testscores2(numericLabels==-1);
 FAR = sum(attack_scores2>threashold2) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold2) / numel(real_scores1)*100;
 HTER2 = (FAR+FRR)/2
[x2,y2,~,AUC2] = perfcurve(numericLabels, testscores2(:,1),1);
AUC2
 plot(x2,y2,'-m','LineWidth',1.8,'MarkerSize',1.8)
 grid on
 hold on
%% Features extraction based on the last average pooling layer of FINE-TUNED model for the training set 
featureLayer = 'avg_pool' ;
imdsLabeled.ReadFcn = @(filename)readAndPreprocessImage(filename);
trainingFeatures1 = activations(CNN2, imdsLabeled, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of  FINE-TUNED model  for the development set 
DevelopmentSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
developmentFeatures1 = activations(CNN2, DevelopmentSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Features extraction based on the last average pooling layer of FINE-TUNED model for the testing set 
% Extract testing set features using the CNN
TestingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures1 = activations(CNN2, TestingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%%

rng(13);
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels';

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
       'Verbose', false);

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
 numericLabels = grp2idx(TestLables);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores3(numericLabels==1);
 attack_scores2 =  testscores3(numericLabels==-1);
 FAR = sum(attack_scores2>threashold3) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold3) / numel(real_scores1)*100;
 HTER3 = (FAR+FRR)/2
[x3,y3,~,AUC3] = perfcurve(numericLabels, testscores3 (:,1),1);
AUC3
plot(x3,y3,'-c','LineWidth',1.8,'MarkerSize',1.8)
grid on
hold off
legend('Supervised learning', 'Semi-Supervised learning','Semi-Supervised + RNN')


clc;clear;

net = alexnet;

imdsTrain = imageDatastore('dataTrain','IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

idmsTest = imageDatastore('dataTest','IncludeSubfolders',true,'LabelSource','foldernames');
idmsTest.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

imdsTrain1 = imageDatastore('dfilTrain','IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrain1.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

idmsTest1 = imageDatastore('dfilTest','IncludeSubfolders',true,'LabelSource','foldernames');
idmsTest1.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MiniBatchSize',8,...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4,...
     'Verbose',false);

%% DATA1 
netTransfer = trainNetwork(imdsTrain,layers,options);
[YPred1,scores1] = classify(netTransfer,idmsTest);
accuracy1 = mean(YPred1 == idmsTest.Labels);

%% DATA2 
netTransfer1 = trainNetwork(imdsTrain1,layers,options);
[YPred2,scores2] = classify(netTransfer1,idmsTest1);
accuracy2 = mean(YPred2 == idmsTest1.Labels);

%% SCORE BASED ALEXNET

pp=scores1+scores2;
for i=1:115
    if pp(i,1)>=pp(i,2)
        YY(i)=1;
    else 
        YY(i)=2;
    end
end
accuracy3 = mean(YY' == double(idmsTest.Labels));


% This code is just a simple feature extraction and classification task on
% IKFDB samples. Features are HOG and SURF and classification is conducted
% by SVM and deep learning CNN
% -Publishing any sample of IKFDB is illegal.
% -Samples inside the database could be used just in a scientific experiment and not for any other purposes.
% If you used samples of this database in your experiment you have to cite it properly as bellow:
% Mousavi, Seyed Muhammad Hossein, and S. Younes Mirinezhad. "Iranian Kinect face database (IKFDB):
% a color-depth based face database collected by Kinect v. 2 sensor." SN Applied Sciences 3.1 (2021): 1-17.
% -In order to use or get full databse, it is required to send me a letter from your supervisor that it 
% is going to be used just for your scientific experiment and responsibility of any other usage is with your supervisor.
% -My Email:    mosavi.a.i.buali@gmail.com 
% -Important: publishing any sample from IKFDB into the web or with any other publishing methods at anywhere is illegal.
% 
%%
clc;
clear;
%% Reading 
path='IKFDB';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);
end;

%% Extract HOG Features
for i = 1 : filesnumber(1,1)
    % The less cell size the more accuracy 
hog{i} = extractHOGFeatures(images{i},'CellSize',[64 64]);
    disp(['Extract HOG :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
    hogfeature(i,:)=hog{i};
    disp(['HOG To Matrix :   ' num2str(i) ]);
end;
HOG=hogfeature;
%% Extract SURF Features 
imset = imageSet('SURF and CNN','recursive'); 
% Create a bag-of-features from the image database
bag = bagOfFeatures(imset,'VocabularySize',100,'PointSelection','Detector');
% Encode the images as new features
surf = encode(bag,imset);
SURF=surf;
% Combining Feature Matrixes
FinalReady=[HOG SURF];
%% Label
sizefinal=size(FinalReady);
sizefinal=sizefinal(1,2);
FinalReady(1:150,sizefinal+1)=1;
FinalReady(151:300,sizefinal+1)=2;
FinalReady(301:450,sizefinal+1)=3;
FinalReady(451:600,sizefinal+1)=4;
FinalReady(601:750,sizefinal+1)=5;
%% Classification
% SVM
lblknn=FinalReady(:,end);
dataknn=FinalReady(:,1:end-1);
tsvm = templateSVM('KernelFunction','polynomial');
svmclass = fitcecoc(dataknn,lblknn,'Learners',tsvm);
svmerror = resubLoss(svmclass);
CVMdl = crossval(svmclass);
genError = kfoldLoss(CVMdl);
% Predict the labels of the training data.
predictedsvm = resubPredict(svmclass);
ct=0;
for i = 1 : filesnumber(1,1)
if lblknn(i) ~= predictedsvm(i)
    ct=ct+1;
end;
end;
% Compute validation accuracy
SVMAccuracy = 100 - ct;
% Plot Confusion Matrix
figure
cmsvm = confusionchart(lblknn,predictedsvm);
cmsvm.Title = 'SVM';
cmsvm.RowSummary = 'row-normalized';
cmsvm.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scoresvm] = resubPredict(svmclass);
diffscoresvm = scoresvm(:,2) - max(scoresvm(:,1),scoresvm(:,3));
[Xsvm,Ysvm,T,~,OPTROCPTsvm,suby,subnames] = perfcurve(lblknn,diffscoresvm,1);
%
figure;
plot(Xsvm,Ysvm)
hold on
plot(OPTROCPTsvm(1),OPTROCPTsvm(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for SVM')
hold off
%
svmsss=size(Xsvm);
svmsss=svmsss(1,1);
mx=min(Xsvm(Xsvm>0));
my=min(Ysvm(Ysvm>0));
Presvm=max(Xsvm)-mx;
Recsvm=max(Ysvm)-my;
disp(['SVM Precision :   ' num2str(Presvm) ]);
disp(['SVM Recall :   ' num2str(Recsvm) ]);

%% Deep Neural Network
% CNN Facial Expressions
deepDatasetPath = fullfile('SURF and CNN');
imds = imageDatastore(deepDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Number of training (less than number of each class)
numTrainFiles = 145;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    % Input image size for instance: 512 512 3
    imageInputLayer([128 128 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % Number of classes
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',128, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');
netmacro = trainNetwork(imdsTrain,layers,options);
YPred = classify(netmacro,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation) *100;
disp(['SVM Classification Accuracy :   ' num2str(SVMAccuracy)]);
disp(['CNN Recognition Accuracy Is =   ' num2str(accuracy)]);
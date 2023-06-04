addpath(genpath('~/workspace/Documents/uni/sc/GitHub/angleLVQtoolbox/'));
addpath(genpath('~/workspace/Documents/uni/sc/matlab/manopt/'));
addpath(genpath('~/workspace/Documents/uni/sc/matlab/summet_code_20140624/'));
% addpath(genpath('~/workspace/Documents/uni/sc/matlab/mmtoolbox2.3/'));

[filePath,name,ext] = fileparts(matlab.desktop.editor.getActiveFilename);
addpath(genpath(sprintf('%s',filePath)));
eval(sprintf('cd %s',filePath));
%% load some demo data and prepare Cross Validation
load fisheriris.mat
Y = grp2idx(species);
X = meas; % just look at 3 dimensions to be able to plot the result
figure(1);
gscatter(X(:,2), X(:,3), species,'rgb','osd');

CrossValIdx = cvpartition(Y,'KFold',5);
%% run angleLVQ and local ALVQ models
reps = 5;
beta = 0.5;dim = 2;reg = 0;% beta=7;dim=3;reg = 0.005;
prepros = cell(CrossValIdx.NumTestSets,1);
% global ALVQ
ALVQ_performance = array2table(nan(CrossValIdx.NumTestSets*reps,4),"VariableNames",{'fold','rep','trainAcc','testAcc'});
ALVQ = cell(CrossValIdx.NumTestSets,reps);
for fold=1:CrossValIdx.NumTestSets
    fprintf('processing fold %i\n',fold);
    % z-score transformation preprocessing
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:),"omitmissing"),'S',std(X(CrossValIdx.training(fold),:),"omitmissing"));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=Y(CrossValIdx.training(fold));
    testLab=Y(CrossValIdx.test(fold));
    for iter=1:reps
        rng(fold*10+iter); % for reproducibility and same initialization of all models
        % train the ALVQ model with parameters above. 
        % Check for more options use "help angleGMLVQ_train"
        [actModel,fval] = angleGMLVQ_train(trainX, trainLab,'testSet',[testX,testLab],'beta',beta,'dim',dim,'regularization',reg,'Display','off');
        ALVQ{fold,iter} = actModel;
        estTrain=angleGMLVQ_classify(trainX,actModel); % confusionmat(trainLab,estTrain)
        estTest =angleGMLVQ_classify(testX ,actModel); % confusionmat(testLab,estTest)
        ALVQ_performance((fold-1)*reps+iter,1:4) = array2table([fold, iter, mean(estTrain==trainLab),mean(estTest==testLab)]);
    end
end
fprintf('ALVQ avg accuracy training & test: %5.4f %4.4f\n',table2array( varfun(@mean,ALVQ_performance(:,3:end))) );
%% building the average model
% Karcher mean implementation from
% https://www.cs.colostate.edu/~vision/summet/
% Ando mean implementation from 
% http://bezout.dm.unipi.it/software/mmtoolbox/
useClusters = 1:3;
avgALVQ_performance = array2table(nan(prod(useClusters)*CrossValIdx.NumTestSets,4),'VariableNames',{'fold','clusters','trainA','testA'});
avgModels = cell(CrossValIdx.NumTestSets,length(useClusters));
actIter = 1;
for fold = 1:CrossValIdx.NumTestSets
    fprintf('processing fold %i\n',fold);
    % z-score transformation preprocessing
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:),"omitmissing"),'S',std(X(CrossValIdx.training(fold),:),"omitmissing"));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=Y(CrossValIdx.training(fold));
    testLab=Y(CrossValIdx.test(fold));
    rng(fold);
    for act = 1:length(useClusters)
        act_cls = useClusters(act);
        actAVGmodel = avg_cluster_MLVQ(ALVQ(fold,:),act_cls);
        avgModels{fold,act} = actAVGmodel{1};
        for j = 1:size(actAVGmodel{1},2)
            actAVGModel = actAVGmodel{1}{j};
            avgPF_estTrain=angleGMLVQ_classify(trainX,actAVGModel); % confusionmat(trainLab,estTrain)
            avgPF_estTest =angleGMLVQ_classify(testX ,actAVGModel); % confusionmat(testLab,estTest)
            avgALVQ_performance(actIter,:) = array2table([fold, act_cls, mean(avgPF_estTrain==trainLab),mean(avgPF_estTest==testLab)]);
            actIter = actIter+1;
        end
    end
end
%% print the performances
fprintf('ALVQ avg accuracy training & test: %5.4f %4.4f\n',table2array( varfun(@mean,ALVQ_performance(:,3:end))) );
disp('average performance of geodesic mean model and 1 cluster');
varfun(@mean, avgALVQ_performance(avgALVQ_performance.clusters==1,3:end) )
fprintf('%4s %8s %6s %6s\n','Fold','clusters','trainA','testA');
for fold = 1:CrossValIdx.NumTestSets
    fprintf('%4i %8i %6.3f %6.3f\n',table2array( avgALVQ_performance(avgALVQ_performance.fold==fold,1:end) )');
end
%% compute the average model over all folds
modelCell = cellfun(@(x) x(1),avgModels(:,1))'; 
rng(2);
allFoldAVGModel = avg_cluster_MLVQ(modelCell, 1);

all_M = mean(cell2mat(arrayfun(@(fold) prepros{fold}.M,1:5,'UniformOutput',false)'));
all_SD= mean(cell2mat(arrayfun(@(fold) prepros{fold}.S,1:5,'UniformOutput',false)'));
allX = bsxfun(@rdivide,bsxfun(@minus,X,all_M),all_SD);

avg_est = angleGMLVQ_classify(allX,allFoldAVGModel{1}{1}); % confusionmat(Y,avg_est)
fprintf('Average accuracy of the geodesic average LVQ model over averages of all folds on the whole data: %.4f\n',mean(avg_est==Y));

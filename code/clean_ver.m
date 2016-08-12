%% This script is used to train the multi-task learning gpr models for memory gap database and test on memory gap database

% feature_p   : features of photos
% feature_sv  : features of viewed sketches
% feature_s1  : features of 1 hour sketches
% feature_s24 : features of 24 hour sketches
% feature_suv : features of unviewed sketches

%% model training
Num = size(feature_p,1); % number of subjects
index = randperm(Num);
index_train = index(1:ceil((Num/3)*2));
index_test = index(ceil((Num/3)*2)+1:end);

% Parameters
PLOT_DATA       = 1;
covfunc_x       = {'covSEiso'};%{'covSEard'};
M               = 10;    % Number of tasks
D               = 1;    % Dimensionality of input spacce
irank           = M; 

for i = 1:size(feature_p,2)
    
    % data preparation
    [xtrain, xtest, ytrain, irank, nx, ind_kf_train, ind_kx_train] = mtl_data(feature_p,feature_sv,...
        feature_s1,feature_s24,feature_suv,index_train, index_test, i, Num);
    data  = {covfunc_x, xtrain, ytrain, M, irank, nx, ind_kf_train, ind_kx_train};
    
    % training
    [logtheta_all deriv_range] = init_mtgp_default(xtrain, covfunc_x, M, irank); % initial the hyper parameters
    [logtheta_all nl]          = learn_mtgp(logtheta_all, deriv_range, data);% learn the models
    
    % testing
    [ Ypred, Vpred ] = predict_mtgp_all_tasks(logtheta_all, data, xtest ); % test the models. Ypred is the predicted sketch, Vpred is variance
    
    KM = numel(index_test);
    
    fYpred1(:,i) = Ypred(1:KM,1);
    fVpred1(:,i) = Vpred(1:KM,1);
    
    fYpred2(:,i) = Ypred(KM+1:KM*2,2);
    fVpred2(:,i) = Vpred(KM+1:KM*2,2);
    
    fYpred3(:,i) = Ypred(KM*2+1:KM*3,3);
    fVpred3(:,i) = Vpred(KM*2+1:KM*3,3);
    
    fYpred4(:,i) = Ypred(KM*3+1:KM*4,4);
    fVpred4(:,i) = Vpred(KM*3+1:KM*4,4);
    
    fYpred5(:,i) = Ypred(KM*4+1:KM*5,5);
    fVpred5(:,i) = Vpred(KM*4+1:KM*5,5);
    
    fYpred6(:,i) = Ypred(KM*5+1:KM*6,6);
    fVpred6(:,i) = Vpred(KM*5+1:KM*6,6);
   
    fYpred7(:,i) = Ypred(KM*6+1:KM*7,7);
    fVpred7(:,i) = Vpred(KM*6+1:KM*7,7);
    
    fYpred8(:,i) = Ypred(KM*7+1:KM*8,8);
    fVpred8(:,i) = Vpred(KM*7+1:KM*8,8);
    
    fYpred9(:,i) = Ypred(KM*8+1:KM*9,9);
    fVpred9(:,i) = Vpred(KM*8+1:KM*9,9);

    fYpred10(:,i) = Ypred(KM*9+1:end,10);
    fVpred10(:,i) = Vpred(KM*9+1:end,10);

end

%% evaluation on memory gap database
task1 = fYpred1;
task2 = fYpred2;
task3 = fYpred3;
task4 = fYpred4;
task5 = fYpred5;
task6 = fYpred6;
task7 = fYpred7;
task8 = fYpred8;
task9 = fYpred9;
task10 = fYpred10;

M = numel(index_test);

dmat1 = cal_dist2(task1,feature_p(index_test,:),fVpred1);
[sortMat,sortIdx] = sort(dmat1,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc1 = cumsum(hist(matchRankNN, 1:M))/M;

dmat2 = cal_dist2(task2,feature_p(index_test,:),fVpred2);
[sortMat,sortIdx] = sort(dmat2,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc2 = cumsum(hist(matchRankNN, 1:M))/M;

dmat3 = cal_dist2(task3,feature_p(index_test,:),fVpred3);
[sortMat,sortIdx] = sort(dmat3,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc3 = cumsum(hist(matchRankNN, 1:M))/M;

dmat4 = cal_dist2(task4,feature_p(index_test,:),fVpred4);
[sortMat,sortIdx] = sort(dmat4,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc4 = cumsum(hist(matchRankNN, 1:M))/M;

dmat5 = cal_dist2(task5,feature_p(index_test,:),fVpred5);
[sortMat,sortIdx] = sort(dmat5,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %
cmc5 = cumsum(hist(matchRankNN, 1:M))/M;

dmat6 = cal_dist2(task6,feature_p(index_test,:),fVpred6);
[sortMat,sortIdx] = sort(dmat6,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc6 = cumsum(hist(matchRankNN, 1:M))/M;

dmat7 = cal_dist2(task7,feature_p(index_test,:),fVpred7);
[sortMat,sortIdx] = sort(dmat7,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc7= cumsum(hist(matchRankNN, 1:M))/M;

dmat8 = cal_dist2(task8,feature_p(index_test,:),fVpred8);
[sortMat,sortIdx] = sort(dmat8,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc8 = cumsum(hist(matchRankNN, 1:M))/M;

dmat9 = cal_dist2(task9,feature_p(index_test,:),fVpred9);
[sortMat,sortIdx] = sort(dmat9,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc9 = cumsum(hist(matchRankNN, 1:M))/M;

dmat10 = cal_dist2(task10,feature_p(index_test,:),fVpred10);
[sortMat,sortIdx] = sort(dmat10,2,'ascend'); %Sort the distance matrix.
[i,matchRankNN]=find(sortIdx==((1:M)'*ones(1,M)));  %Find the rank of the true match. Pefmect is all 1s.
cmc10 = cumsum(hist(matchRankNN, 1:M))/M;

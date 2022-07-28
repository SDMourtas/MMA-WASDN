%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  A 4-layer feed-forward neuronet model, trained by a MA-WASD      %
%  algorithm. (version 1.0)                                         %
%                                                                   %
%  Developed in MATLAB R2018b                                       %
%                                                                   %
%  Author and programmer: T.E. Simos, V.N.Katsikis, S.D.Mourtas     %
%                                                                   %
%   e-Mail: tsimos.conf@gmail.com                                   %
%           vaskatsikis@econ.uoa.gr                                 %
%           spirosmourtas@gmail.com                                 %
%                                                                   %
%   Main paper: T.E. Simos, S.D.Mourtas, V.N.Katsikis,              %
%               "A Multi-Input with Multi-Function Activated WASD   %
%               Neuronet for Classification Problems with           %
%               Applications in Firm Fraud and Loan Approval",      %
%               Applied Soft Computing, 109351 (2022)               %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
clc

% Choose modeling problem (for x = 1 to 3)
x=1; 
[X_train,Y_train,X_test,Y_test,p,dmax]=problem(x);
[G,M]=size(X_train);

%% Training
% Data Preprocessing
[X_N,X_min,X_max]=Normalization(X_train); % Normalization
[Y_N,~,~]=Normalization(Y_train);

% Neuronet model
tic
[W,Em,N,c,E]=MA_WASD(X_N,Y_N,G,M,p,dmax);  % MMA-WASDN's optimal hidden-layer structure
toc
tic
[W_MIHPN,N_MIHPN]=MIHPN_WASD(X_N,Y_N,G,M); % MIHPN_WASDN's optimal hidden-layer structure
toc
tic
FineKNN_Model=FineKNN_Cl([X_train,Y_train],x);     % Fine KNN model 
toc
tic
FineTree_Model=FineTree_Cl([X_train,Y_train],x);   % Tree: Fine Tree model 
toc
tic
LinearSVM_Model=LinearSVM_Cl([X_train,Y_train],x); % Linear SVM model 
toc
tic
EBT_Model=EBT_Cl([X_train,Y_train],x);             % Ensemble Bagged Trees model 
toc
tic
KNB_Model=KNBayes_CL([X_train,Y_train],x);         % Kernen Naive Bayes model 
toc

%% Predict
pred=predictN(X_test,M,W,N,c,X_min,X_max,1);       % MMA-WASDN prediction
pred_MIHPN=predictN(X_test,M,W_MIHPN,N_MIHPN,c,X_min,X_max,2); % MIHPN_WASDN prediction
predKNN = FineKNN_Model.predictFcn(X_test);        % KNN prediction
predFT = FineTree_Model.predictFcn(X_test);        % FT prediction
predSVM = LinearSVM_Model.predictFcn(X_test);      % SVM prediction
predEBT = EBT_Model.predictFcn(X_test);            % EBT prediction
predKNB = KNB_Model.predictFcn(X_test);            % KNB prediction

fprintf('MMA-WASDN statistics: \n');error_pred(pred,Y_test); % Error of test data
fprintf('MIHPN-WASDN statistics: \n');error_pred(pred_MIHPN,Y_test);
fprintf('KNN statistics: \n');error_pred(predKNN,Y_test);
fprintf('FT statistics: \n');error_pred(predFT,Y_test);
fprintf('SVM statistics: \n');error_pred(predSVM,Y_test);
fprintf('EBT statistics: \n');error_pred(predEBT,Y_test);
fprintf('KNB statistics: \n');error_pred(predKNB,Y_test);

[h,p] = McNemar_test([pred,pred_MIHPN,predKNN,predFT,predSVM,predEBT,predKNB],Y_test);

%% Figures
Problem_figures(pred,Y_test,E,N,c)

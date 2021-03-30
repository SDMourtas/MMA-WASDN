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
%              (submitted)                                          %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
clc

% Choose modeling problem (for x = 1 to 3)
x=1; 
[X_train,Y_train,X_test,Y_test,p]=problem(x);
[G,M]=size(X_train);
dmax=40; % maximum number of hidden-layer

%% Training
% Data Preprocessing
[X_N,X_min,X_max]=Normalization(X_train); % Normalization
[Y_N,~,~]=Normalization(Y_train);

% Neuronet model
tic
[W,Em,N,c,E]=MA_WASD(X_N,Y_N,G,M,p,dmax); % optimal hidden-layer structure
toc

%% Predict
pred=predictN(X_test,M,W,N,c,X_min,X_max); % data prediction

Et=error_pred(pred,Y_test); % Error of test data

%% Figures
Problem_figures(pred,Y_test,E,N,c)
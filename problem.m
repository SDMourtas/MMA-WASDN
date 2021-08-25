function [X_train,Y_train,X_test,Y_test,p,dmax]=problem(xx)
% Input and preparation of the training and testing data for the MMA-WASDN model
warning off
if xx==1
    filename='audit_data.csv';  % Example 1
    data=readtable(filename,'ReadVariableNames',true,'ReadRowNames' ,false,'TreatAsEmpty' ,'-');
    data=data(:,[end 1:end-1]);
    [x,~]=data_prep(data); % Preprocessing Data: change strings with numbers
    varnames=data.Properties.VariableNames;
    T = array2table(x,'VariableNames',varnames);
    p=0.8;
    dmax=10; % maximum number of hidden-layer
elseif xx==2
    filename='cs-training.csv';  % Example 2
    data=readtable(filename,'ReadVariableNames',true,'ReadRowNames' ,true,'TreatAsEmpty' ,'-');
    [x,~]=data_prep(data); % Preprocessing Data: change strings with numbers
    varnames=data.Properties.VariableNames;
    T = array2table(x,'VariableNames',varnames);
    p=0.85;
    dmax=5; % maximum number of hidden-layer
elseif xx==3
    filename='SBAcase.11.13.17.csv';  % Example 3
    data=readtable(filename,'ReadVariableNames',true,'ReadRowNames' ,false,'TreatAsEmpty' ,'-');
    data=data(:,[end 1:20 22:end-1]);
    [x,~]=data_prep(data); % Preprocessing Data: change strings with numbers
    varnames=data.Properties.VariableNames;
    T = array2table(x,'VariableNames',varnames);
    p=0.75;
    dmax=5; % maximum number of hidden-layer
else
    fprintf('Error: No valid problem number.\n')
    return
end

% remove rows which include Nan in their columns
col=2:size(T,2);
[row,~]=find(isnan(T{:,col}));
row=unique(row);q=1:size(T,1);q(row)=[];T=T(q,:);

% training-testing data
if xx==2
    q1=find(T{:,1}==0);q2=find(T{:,1}==1);
    q3=[q1(1:1e4)' q2'];
    train_row=q3(1:2:end); test_row=[q3(2:2:end-1) q1(1e4+1:end)']; 
else
    train_row=1:2:size(T,1); test_row=2:2:size(T,1)-1; 
end
X_train=T{train_row,col}; Y_train=T{train_row,1};
X_test=T{test_row,col}; Y_test=T{test_row,1};
function Y=predictN(X,M,W,N,c,x_min,x_max,y)
% function for predicting

x_N=Normalization(X,x_min,x_max);

if y==1
    Q=Qmatrix(x_N,M,size(X,1),N,c);
elseif y==2
    Q=Qmatrix2(x_N,M,size(X,1),N);
end
Y_N=Q*W; 

Y=zeros(size(Y_N));
Y(Y_N>=-0.375)=1;
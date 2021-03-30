function [W,Em,N,c,E]=MA_WASD(X,Y,G,M,p,dmax)
% function for finding the optimal hidden-layer neurons weights of the
% neuronet

% Initialization
N=[]; c=[];% the neurons number of the hidden layer (i.e., hidneurons)
Em=inf; E=zeros(dmax+1,1); Ev=zeros(4,1);
G1=round(p*G); % size of data fitting
G2=G-G1; % size of data validation

for d=0:dmax
% WDD Method
for i=1:4
Q=Qmatrix(X,M,G,[N;d],c,i);
W=pinv(Q(1:G1,:))*Y(1:G1); 
Ev(i)=100/G2*sum(abs(Q(G1+1:G,:)*W-Y(G1+1:G))); % MAE
end
E(d+1)=min(Ev);
if E(d+1)<Em
    r=find(Ev==min(Ev));
    Em=E(d+1);N=[N;d];c=[c;r(1)];
end
end

% output
Q=Qmatrix(X,M,G,N,c);
W=pinv(Q)*Y; 
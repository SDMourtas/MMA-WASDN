function E=error_pred(X,Y)

R=X-Y;
[row,col]=find(isnan(R)); R(row,col)=0;

E=100/length(R)*sum(abs(R)); % MAE
fprintf('The test data MAE is: %f %%\n',E)
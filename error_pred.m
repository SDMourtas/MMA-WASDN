function E=error_pred(X,Y)

R=X-Y; T=length(Y);
E=sum(abs(R))/T; % MAE
fprintf('The test data MAE is: %f \n',E)

Y2=find(Y==1); 
R2=sum(X(Y2)==1)/length(Y2); % TP: true positive
fprintf('The TP of the model on the test data is: %f \n',R2)

R3=sum(X(Y2)==0)/length(Y2); % FP: false positive
fprintf('The FP of the model on the test data is: %f \n',R3)

Y3=find(Y==0); 
R4=sum(X(Y3)==0)/length(Y3); % TN: true negative
fprintf('The TN of the model on the test data is: %f \n',R4)

R5=sum(X(Y3)==1)/length(Y3); % FN: false negative
fprintf('The FN of the model on the test data is: %f \n',R5)

prec=R2/(R2+R3); rec=R2/(R2+R5);
fprintf('The Precision of the model on the test data is: %f \n',prec)
fprintf('The Recal of the model on the test data is: %f \n',rec)

R1=sum(X==Y)/T; % accuracy
fprintf('The accuracy of the model on the test data is: %f \n',R1)

Fm=2*prec*rec/(prec+rec); % F-measure
fprintf('The F-measure of the model on the test data is: %f \n',Fm)
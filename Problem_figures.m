function Problem_figures(pred,Y_test,E,N,c)
[k,ind]=sort(Y_test);
k_pr=pred(ind);
k_def=find(k==1);

figure
b1=k_def(1)-1;      % actual false, "0"
b3=sum(k_pr(1:b1)); % predicted true, "1", in comparison with the actual false
b2=b1-b3;           % predicted false, "0", in comparison with the actual false
b4=sum(k);          % actual true, "1"
b5=sum(k_pr(b1+1:end)); % predicted true, "1", in comparison with the actual true
b6=b4-b5;           % predicted false, "0", in comparison with the actual true
b=[b1 b2 b3;b4 b5 b6];
Bar=bar([1 2],b);
ctr=zeros(3,2);ydt=ctr;
for i = 1:3
    ctr(i,:)=bsxfun(@plus, Bar(i).XData, Bar(i).XOffset'); 
    ydt(i,:)=Bar(i).YData;                                  
end
for i=1:2
    text(ctr(:,i),ydt(:,i),sprintfc('%d',ydt(:,i)),...
        'horiz','center', 'vert','bottom')
end
box off
ylabel('Test Samples')
xlabel('Classification')
xticklabels({'True','False'})
legend('Actual Clasiffication','Correct Predictions','Incorrect Predictions')

figure
semilogy(0:length(E)-1,E,'DisplayName','Error of Optimal N')
hold on
if any(c==1)
    v=find(c==1);
    semilogy(N(v),E(N(v)+1),'.','Color',[0.8500 0.3250 0.0980],...
        'MarkerSize',16,'DisplayName','AF: Power')
end
if any(c==2)
    v=find(c==2);
    semilogy(N(v),E(N(v)+1),'.','Color',[0.4660 0.6740 0.1880],...
        'MarkerSize',16,'DisplayName','AF: Power Sigmoid')
end
if any(c==3)
    v=find(c==3);
    semilogy(N(v),E(N(v)+1),'.','Color',[0.4940 0.1840 0.5560],...
        'MarkerSize',16,'DisplayName','AF: Power Inv. Exp.')
end
if any(c==4)
    v=find(c==4);
    semilogy(N(v),E(N(v)+1),'.','Color',[0.9290 0.6940 0.1250],...
        'MarkerSize',16,'DisplayName','AF: Power Softplus')
end
xlabel('N');
ylabel('MAE %')
legend
hold off
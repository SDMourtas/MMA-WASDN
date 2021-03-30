function [x,a]=data_prep(data)
% Preprocessing Data: change strings with numbers
[m,n]=size(data);
x=zeros(m,n);k=0;
for j=1:n
    if isreal(data{1,j})
        x(:,j)=data{:,j};
    elseif isdatetime(data{1,j})
        x(:,j)=datenum(data{:,j});
    else
        b=str2double(data{1,j});
        if ~isnan(b)
            x(:,j)=str2double(data{:,j});
        else
            k=k+1;r=1;x(1,j)=r;
            a.word(k)=data{1,j};a.replaced(k)=r;a.column(k)=j;
            for i=2:m
                b=find(strcmp(data{i,j},a.word));
                if ~isempty(b) 
                    if any(j==a.column(b))
                        x(i,j)=a.replaced(b(j==a.column(b)));  
                    else
                        k=k+1;r=r+1;
                        a.word(k)=data{i,j};
                        a.replaced(k)=r;
                        a.column(k)=j;
                        x(i,j)=r;                          
                    end
                else 
                    k=k+1;r=r+1;
                    a.word(k)=data{i,j};
                    a.replaced(k)=r;
                    a.column(k)=j;
                    x(i,j)=r;                 
                end
            end
        end
    end
end
if k==0
    a=0;
end
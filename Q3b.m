clear all
clc
close all 
X = randn(800,2);
s2 = sum(X.^2,2);
trainX = (X.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';
plot(trainX(1,:),trainX(2,:),'+r');
somRow = 8;
somCol = 8;
sigma0 = sqrt(2^2+2^2)/2;
iter =550;

weights = -1+2*rand(somRow,somCol,2);
shaped_w =reshape(weights,2,somRow*somCol);
%lr = 0.1;
k =1;
for epoch = 0:iter
    % plotting the SOM
    if(rem(epoch,50) == 0|epoch ==1|epoch ==2|epoch ==4)
          figure(2)
    plot(trainX(1,:),trainX(2,:),'+r'); axis equal    
    hold on
    plot(shaped_w(1,:),shaped_w(2,:),'bo','Linewidth',1.5)
    hold off                
    title(["Self Organizing Map for Circle at the iteration:",epoch]);
    filename ="Q3b" +k;
    xlabel("x");
    ylabel("y");
    saveas(gcf,filename,"png");
    k = k+1;
    end
    for i = 1:size(trainX,2)
        shaped_w =reshape(weights,2,somRow*somCol);
        WMinusX = shaped_w-repmat(trainX(:,i),1,somRow*somCol);
        squared= WMinusX.^2;
        distance = (squared(1,:)+squared(2,:)).^0.5;
        winner = find(distance == min(distance));
        winCol = rem(winner,somRow);
        winRow = (winner-winCol)/somCol;
        for j = 1:size(shaped_w,2)
            Col = rem(j,somRow);
            Row = (j-Col)/somCol;
            %fprintf("Epoch %d Input %d Neuron %d ,negh result %f \n",epoch,i,j,neighborhood([1,j],[1,winner],M,N,j));
            shaped_w(:,j) = shaped_w(:,j)-learning_rate(epoch,iter)*neighborhood([Row,Col],[winRow,winCol],sigma0,epoch,iter)*(WMinusX(:,j));
        end
        weights = reshape(shaped_w,somRow,somCol,2);
    end
    
 	
    
  
end

  

function h  = neighborhood(w1,w2,sigma0,n,iter)
    T = iter;
    Tau1 = T/log(sigma0);
    sigma = sigma0 *exp(-n/Tau1);
    d = norm(w1-w2);
    h = exp(-d^2/(2*sigma^2));
end

function lr  = learning_rate(n,iter)
   lr0 = 0.1;
   tau =iter;
   lr = lr0*exp(-n/tau);
end
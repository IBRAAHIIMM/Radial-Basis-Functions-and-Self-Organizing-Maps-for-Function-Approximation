clear all
clc
close all 
x = linspace(-pi,pi,400);
y = sin(pi.*x)/(pi.*x);
M = 40;
N = 1;
sigma0 = sqrt(4^2+4^2)/2;
iter =550;
for i = 1:size(x,2)
    if(x(i) == 0)
        y(1,i) = 1;
    else
        y(1,i) = sin(pi*x(1,i))/(pi*x(1,i));
    end
end

trainX = [x;y];
figure(1)
plot(trainX(1,:),trainX(2,:),'+r'); axis equal
weights = -4+8*rand(2,M*N);
%lr = 0.1;
k=1;
for epoch = 0:iter
    for i = 1:size(trainX,2)
        WMinusX = weights-repmat(trainX(:,i),1,M);
        squared= WMinusX.^2;
        distance = (squared(1,:)+squared(2,:)).^0.5;
        winner = find(distance == min(distance));
        for j = 1:size(weights,2)
            %fprintf("Epoch %d Input %d Neuron %d ,negh result %f \n",epoch,i,j,neighborhood([1,j],[1,winner],M,N,j));
            weights(:,j) = weights(:,j)-learning_rate(epoch,iter)*neighborhood([1,j],[1,winner],sigma0,epoch,iter)*(WMinusX(:,j));
        end
    end
 	             % plotting the SOM
    if(rem(epoch,50) == 0)
          figure(2)
        plot(trainX(1,:),trainX(2,:),'+r'); axis equal    
        hold on% plotting the "hat"
        plot(weights(1,:),weights(2,:),'-bo','Linewidth',1.5)
        hold off% plotting random w.         
        title(["Self Organizing Map for Sinc function in the iteration",epoch]);
        filename ="Q3a" +k;
        xlabel("x");
        ylabel("y");
        saveas(gcf,filename,"png");
        k = k+1;
       % plot(w(1,:),w(2,:),'k-o','Linewidth',1.5) 	
    end
end

  

function h  = neighborhood(w1,w2,sigma0,n,iter)
    T = iter;
    Tau1 = T/log(sigma0);
    sigma = sigma0 *exp(-n/Tau1);
    d = sum((w1-w2).^2);
    h = exp(-d^2/(2*sigma^2));
end

function lr  = learning_rate(n,iter)
   lr0 = 0.1;
   tau =iter;
   lr = lr0*exp(-n/tau);
end
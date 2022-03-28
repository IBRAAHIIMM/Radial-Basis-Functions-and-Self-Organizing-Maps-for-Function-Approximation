%LAB 03 Question 1
%RBF is Gaussian with mean = 0, standart deviation = 0.1
x_train = -1.6:0.08:1.6;
y_train = 1.2*sin(pi*x_train)-cos(2.4*pi*x_train)+0.3*randn(1,41);
x_test  = -1.6:0.01:1.6;
y_test = 1.2*sin(pi*x_test)-cos(2.4*pi*x_test);

standDev = 0.1;
centers = x_train;
interpolation = zeros(size(centers,2),size(centers,2));
row = 1;
%calculate interpolation matrix
for i=x_train
    r = centers-i;
    rowPhi = RBF(r,standDev);
    interpolation(row,:) = rowPhi;
    row = row+1;
end
weights = inv(interpolation)*transpose(y_train);

interpolation_test = repmat(centers,size(x_test,2),1)-repmat(transpose(x_test),1,size(centers,2));
y_out = RBF(interpolation_test,standDev)*weights;

figure(1)
plot(x_test,y_test,x_test,y_out);
xlabel("x");
ylabel("y");
legend(["Function","Network Output"]);
title("Exact interpolation method for approximating y(x)");
saveas(gcf,"Q1a","png");

function out = RBF(r,std)
    out = exp(-r.^2/(2*std^2));
end
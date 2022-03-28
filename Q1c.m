%LAB 03 Question 1
%RBF is Gaussian with mean = 0, standart deviation = 0.1
x_train = -1.6:0.08:1.6;
y_train = 1.2*sin(pi*x_train)-cos(2.4*pi*x_train)+0.3*randn(1,41);
x_test  = -1.6:0.01:1.6;
y_test = 1.2*sin(pi*x_test)-cos(2.4*pi*x_test);
regularization = 0:0.05:0.6;
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
I = eye(size(interpolation));
i =1;
for reg = regularization
    weights = inv(interpolation.'*interpolation+reg*I)*interpolation.'*y_train.';
    interpolation_test = repmat(centers,size(x_test,2),1)-repmat(transpose(x_test),1,size(centers,2));
    y_out = RBF(interpolation_test,standDev)*weights;
    figure(i)
    plot(x_test,y_test,x_test,y_out);
    xlabel("x");
    ylabel("y");
    legend(["Function","Network Output"]);
    title(["Exact interpolation method with regularization parameter,",reg," for approximating y(x)"]);
    filename ="Q1creg" +i;
    saveas(gcf,filename,"png");
    i = i+1;
end



function out = RBF(r,std)
    out = exp(-r.^2/(2*std^2));
end
%LAB 03 Question b
%RBF with Fixed Centers Selected at Random
x_train = -1.6:0.08:1.6;
y_train = 1.2*sin(pi*x_train)-cos(2.4*pi*x_train)+0.3*randn(1,41);
x_test  = -1.6:0.01:1.6;
y_test = 1.2*sin(pi*x_test)-cos(2.4*pi*x_test);
M = 20;
r = randi([1 41],1,20);
centers = x_train(r);
dmax = max(centers)-min(centers);
spreads = dmax/(2*M)^0.5;

interpolation = zeros(size(x_train,2),size(centers,2));
row = 1;
for x = x_train
    interpolation(row,:) = RBF(x,centers,spreads);
    row = row+1;
end
%weights = inv(interpolation.'*interpolation)*interpolation.'*y_train.';
%weights = pinv(interpolation)*y_train.';
weights = pinv(interpolation.'*interpolation)*interpolation.'*y_train.';

row = 1;
interpolation_test= zeros(size(x_test,2),size(centers,2));
for x = x_test
    interpolation_test(row,:) = RBF(x,centers,spreads);
    row = row+1;
end

y_out =interpolation_test*weights;

figure(1)
plot(x_test,y_test,x_test.',y_out);
xlabel("x");
ylabel("y");
title("Fixed Centers Selected at Random method for approximating y(x)");
legend(["Function","Network Output"]);
saveas(gcf,"Q1b","png");

function out = RBF(x,center,std)
    out = exp(-(x-center).^2/(2*std^2));
end
clc 
clear all 
load("mnist_m.mat");
%tmp=reshape(train_data(:,column_no),28,28);
%   imshow(tmp);
trainIdx = find(train_classlabel==2 | train_classlabel==2);
train_class1 = train_data(:,train_classlabel== 2);
train_class1 = [train_class1,train_data(:,train_classlabel== 9)];
train_label1 = ones(199,1);

train_class0 = train_data(:,(train_classlabel~= 2)&(train_classlabel~= 9));
train_label0 = zeros(801,1);

test_class1 = test_data(:,test_classlabel== 2);
test_class1 = [test_class1,test_data(:,test_classlabel== 9)];
test_label1 = ones(51,1);

test_class0 = test_data(:,(test_classlabel~= 2)&(test_classlabel~= 9));
test_label0 = zeros(199,1);

TrLabel = [train_label0;train_label1];
TrData = [train_class0,train_class1];

TeLabel = [test_label0;test_label1];
TeData = [test_class0,test_class1];

N = 1000; %Number of samples = Number of hidden neurons = N
centers = TrData;
std= 100;
interpolation =zeros(N,N);
for i = 1:N
    for j =1:N
         squared = (centers(:,j)-TrData(:,i)).^2;
         r = sum(squared)^0.5;
         interpolation(i,j) = exp(-r^2/(2*std^2));
    end 
end

regularization = 0:0.1:1.1;
accuracy_data = zeros(1,size(regularization,2));
I = eye(size(interpolation));
k =1;
for reg =regularization
    weights = inv(interpolation.'*interpolation+reg*I)*interpolation.'*TrLabel;
    interpolation_test =zeros(size(TeData,2),N);
    for i = 1:size(TeData,2)
        for j =1:N
             squared = (centers(:,j)-TeData(:,i)).^2;
             r = sum(squared)^0.5;
             interpolation_test(i,j) = exp(-r^2/(2*std^2));
        end 
    end
    TePred = interpolation_test*weights;
    interpolation_train =zeros(size(TrData,2),N);
    for i = 1:size(TeData,2)
        for j =1:N
             squared = (centers(:,j)-TrData(:,i)).^2;
             r = sum(squared)^0.5;
             interpolation_train(i,j) = exp(-r^2/(2*std^2));
        end 
    end
    TrPred = interpolation_train*weights;
    
    thr = zeros(1,1000);  %threshold for each 
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    figure(k)
    plot(thr,TrAcc,'-b ',thr,TeAcc,'-r');legend('tr','te');
    title(["Exact interpolation method with regularization parameter,",reg]);
    filename ="Q2areg" +k;
    xlabel("Threshold");
    ylabel("Accuracy");
    saveas(gcf,filename,"png");
    %accuracy_data(1,k) = matching(1,2)/size(TeLabel,1)*100; %percentage accuracy of RBFN with specific regularization
    k = k+1;
end




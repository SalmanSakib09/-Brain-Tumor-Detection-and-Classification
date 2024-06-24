% Evaluate performance based on parameters
function svmStruct_Latest = crossfun_Parameters(xtrain, ytrain, xtest, rbf_sigma, boxconstraint)
    svmStruct = fitcsvm(xtrain, ytrain, 'KernelFunction', 'RBF', 'KernelScale', rbf_sigma, 'BoxConstraint', boxconstraint);
    yfit = predict(svmStruct, xtest);

c = cvpartition(200,'kfold',10);
minfn = @(z)crossval('mcr',cdata,grp,'Predfun', ...
    @(xtrain,ytrain,xtest)crossfun(xtrain,ytrain,...
    xtest,exp(z(1)),exp(z(2))),'partition',c);
opts = optimset('TolX',5e-4,'TolFun',5e-4);

[searchmin fval] = fminsearch(minfn,randn(2,1),opts);
[searchmin fval] = fminsearch(minfn,randn(2,1),opts);
[searchmin fval] = fminsearch(minfn,randn(2,1),opts);
z = exp(searchmin)
svmStruct_Latest = fitcsvm(cdata,grp,'KernelFunction','rbf','rbf_sigma',z(1),'boxconstraint',z(2));
end
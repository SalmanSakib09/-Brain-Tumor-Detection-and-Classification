% Cross Validation
function yfit = crossfun(xtrain,ytrain,xtest)
svmStruct = fitcsvm(xtrain, ytrain, 'KernelFunction', 'RBF', 'KernelScale', 'BoxConstraint');
yfit = predict(svmStruct,xtest);
c = cvpartition(200,'kfold',10);
minfn = @(z)crossval()
end

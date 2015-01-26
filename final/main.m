import model.classify.SoftMarginClassifier;
import model.classify.KernelClassifier;
import model.classify.LaplacianSupportVectorMachine;

clear;

load('data/X.mat');
load('data/y.mat');
load('data/Xtest.mat');
load('data/ytest.mat');


X = zscore(X);
pcaval = pca(X);
X = X * pcaval;
%X = X*pca(X);



Xtest = zscore(Xtest);
Xtest = Xtest*pcaval;
%Xtest = Xtest*pca(Xtest);



%[Xtest, ~] = compute_mapping(Xtest, 'LDA');

sv = LaplacianSupportVectorMachine.train(X, y);
out = sv.predict(X);

%svm = svmtrain(out, X, '-t 0');

%svm = SoftMarginClassifier.train(X*k, out);

%lbl = svm.predict(Xtest*k2);
%1-(nnz(lbl - ytest)/length(ytest))

sv = KernelClassifier.train(X, out);
%out = sv.predict(X);

%sv2 = KernelClassifier.train(X, out);

lbl = sv.predict(Xtest);

1-(nnz(lbl - ytest) / length(ytest))


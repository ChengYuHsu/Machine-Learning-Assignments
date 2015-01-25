import model.classify.SoftMarginClassifier;

load('data/X.mat');
load('data/y.mat');
load('data/Xtest.mat');
load('data/ytest.mat');

%X = X(:,1:3);
X = zscore(X);

%Xtest = Xtest(:,1:3);
Xtest = zscore(Xtest);

%[Xtest, ~] = compute_mapping(Xtest, 'LDA');

options=make_options('gamma_I', 1,'gamma_A', 1e-5,'NN', 3,'KernelParam', 0.55);
options.Verbose=1;
options.UseBias=1;
options.UseHinge=1;
options.LaplacianNormalize=0;
options.NewtonLineSearch=0;


data.X=X;
data.Y=y;

data.K=calckernel(options, X);
data.L=laplacian(options, X);

classifier=lapsvmp(options,data);

fprintf('It took %f seconds.\n',classifier.traintime);
out=sign(data.K(:,classifier.svs)*classifier.alpha+classifier.b);

%svm = svmtrain(out, X, '-t 0');

svm = SoftMarginClassifier.train(X, out);
lbl = svm.predict(Xtest);
lbl - ytest
sum(abs(lbl - ytest)./2)



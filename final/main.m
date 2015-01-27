clear;

import model.classify.MLFinalClassifier;

load('data/X.mat');
load('data/y.mat');
load('data/Xtest.mat');
load('data/ytest.mat');
load('data/X_finalds.mat');
load('data/y_finalds.mat');


classifier = MLFinalClassifier.train(X, y);

labels = classifier.predict(X);
acc = 1-nnz(y(y~=0) - labels(y~=0))/length(y(y~=0));
acc

labels = classifier.predict(Xtest);
acc = 1-nnz(labels - ytest)/length(ytest);
acc

labels = classifier.predict(X_finalds);
acc = 1-nnz(labels - y_finalds)/length(y_finalds);
acc




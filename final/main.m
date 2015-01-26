clear;

import model.classify.MLFinalClassifier;

load('data/X.mat');
load('data/y.mat');
load('data/Xtest.mat');
load('data/ytest.mat');
load('data/X_finalds.mat');
load('data/y_finalds.mat');


classifier = MLFinalClassifier.train(X, y);
labels = classifier.predict(Xtest);

acc = 1-nnz(labels - ytest)/length(ytest);


acc

labels = classifier.predict(X_finalds);

acc = 1-nnz(labels - y_finalds)/length(y_finalds);

acc




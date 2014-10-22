import model.classifier.PerceptronClassifier

clear %clear workspace

%--- TODO: please import the dataset here ---%
X = importdata('./data/training/100/X.dat');
y = importdata('./data/training/100/y.dat');



%--- TODO: modify the DummyRegressor to your LinearRegressor & LinearRegressorLocalWeight ---%
%---       please follow the specs strickly                              ---%

lifted_x_2 = [X, X.*X];
lifted_x_3 = [X, X.^2, X.^3];

 myClassifier = PerceptronClassifier.train(X, y);
 value = myClassifier.predict(X);

% myRegressor = LinearRegressorLocalWeight.train(X,y);
% value = myRegressor.predict(X, containers.Map({'tau'}, {100}));

%%% plot data %%%
[N, d] = size(X);

 %scatter (X_testing(y_testing==1), zeros(size(X_testing(y_testing==1))), 'g');
scatter (X(y==1), zeros(size(X(y==1))), 'g');
hold on;
% scatter (X_testing(y_testing==-1), zeros(size(X_testing(y_testing==-1))), 'r');
scatter (X(y==-1), zeros(size(X(y==-1))), 'r');
hold on;


%--- TODO: plot the regressor you train ---%

line_space = [0:0.01:0.3];
 %plot(line_space, myClassifier.w(1)*line_space+myClassifier.w(2)*line_space.^2+myClassifier.w(3)*line_space.^3+myClassifier.b, 'b');
 plot(line_space, myClassifier.w*line_space+myClassifier.b, 'b');
% scatter(X, value, 'r');

hold off;

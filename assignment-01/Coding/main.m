import model.regressor.LinearRegressor
import model.regressor.LinearRegressorLocalWeight;

clear %clear workspace

%--- TODO: please import the dataset here ---%
X = importdata('./data/X.dat');
y = importdata('./data/y.dat');

%--- TODO: modify the DummyRegressor to your LinearRegressor & LinearRegressorLocalWeight ---%
%---       please follow the specs strickly                              ---%

myRegressor = LinearRegressor.train(X,y);
value = myRegressor.predict(X);

% myRegressor = LinearRegressorLocalWeight.train(X,y);
% value = myRegressor.predict(X, containers.Map({'tau'}, {100}));

%%% plot data %%%
scatter (X,y,'g');
hold on;

%--- TODO: plot the regressor you train ---%

plot(X, value, 'r');

% scatter(X, value, 'r');

hold off;

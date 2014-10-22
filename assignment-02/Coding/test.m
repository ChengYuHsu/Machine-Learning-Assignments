
X_testing = importdata('./data/testing/X.dat');
y_testing = importdata('./data/testing/y.dat');

value = myClassifier.predict(X_testing);

scatter (X_testing(y_testing==1), zeros(size(X_testing(y_testing==1))), 'g');

hold on;
 scatter (X_testing(y_testing==-1), zeros(size(X_testing(y_testing==-1))), 'r');

hold on;


%--- TODO: plot the regressor you train ---%

    line_space = [0:0.01:0.3];
    plot(line_space, myClassifier.w*line_space+myClassifier.b, 'b');
 %plot(line_space, myClassifier.w(1)*line_space+myClassifier.w(2)*line_space.^2+myClassifier.w(3)*line_space.^3+myClassifier.b, 'b');
 
 output = y_testing .* value;
 false = length(output(output==-1))
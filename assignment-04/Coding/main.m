import model.classify.SoftMarginLinearClassifier

clear;

X = importdata('data/X.txt');
y = importdata('data/y.txt');

rand50 = randsample(200, 50);
X50 = X(rand50,:);
y50 = y(rand50,:);

rand100 = randsample(200, 100);
X100 = X(rand100,:);
y100 = y(rand100,:);

rand150 = randsample(200, 150);
X150 = X(rand150,:);
y150 = y(rand150,:);

s = cputime;
classifier = SoftMarginLinearClassifier.train(X50, y50);
e = cputime;
e - s

s = cputime;
classifier = SoftMarginLinearClassifier.train(X100, y100);
e = cputime;
e - s

s = cputime;
classifier = SoftMarginLinearClassifier.train(X150, y150);
e = cputime;
e - s

s = cputime;
classifier = SoftMarginLinearClassifier.train(X, y);
e = cputime;
e - s

hold on;
line_x = [-40, 0];
line_y = (classifier.b-classifier.w(1, 1)*line_x)/classifier.w(2, 1);

plot(line_x, line_y, 'red');

scatter(X(y==1,1), X(y==1,2), 'o', 'green');
scatter(X(y==-1,1), X(y==-1,2), 'o', 'blue');

hold off;




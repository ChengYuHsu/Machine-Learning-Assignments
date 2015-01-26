dim1 = 1;
dim2 = 2;
dim3 = 3;

scatter(X(out==1,dim1), X(out==1,dim2), 'r');
hold on;
scatter(X(out==-1,dim1), X(out==-1,dim2), 'b');




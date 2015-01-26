dim1 = 1;
dim2 = 2;
dim3 = 3;

scatter(dat(out==1,:), zeros(length(dat(out==1,:)), 1), 'r');
hold on;
scatter(dat(out==-1,:), ones(length(dat(out==-1,:)), 1), 'b');
hold off;




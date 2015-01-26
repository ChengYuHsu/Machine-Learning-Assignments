classdef KernelClassifier < handle

    properties
        X
        y
        c
        k
    end
    
    methods
        function kernelClassifierObj = KernelClassifier(X, y, k, c)
            kernelClassifierObj.X = X;
            kernelClassifierObj.y = y;
            kernelClassifierObj.k = k;
            kernelClassifierObj.c = c;
        end
        
        function labels = predict (obj, data)
            sigma = 0.5;
            
            [n, ~] = size(obj.X);
            [m, d] = size(data);
            points = zeros(m+n, d);
            
            points(1:n,:) = obj.X;
            points(n+1:m+n,:) = data;
            
            kernel = exp(-1*(squareform(pdist(points)).^2)./(2*(sigma^2)));
            kernel = kernel(n+1:n+m, 1:n);
            
            labels = sign(kernel * obj.c);
            
        end
    end
    
    methods(Static)
        function kernelClassifierObj = train(X, y)
            
            [n, ~] = size(X);
            
            sigma = 0.05;
            lambda = 2;
            kernel = exp(-1*(squareform(pdist(X)).^2)./(2*(sigma^2)));
            
            cvx_begin 
                variable C(n, 1)
                minimize (sum((y - kernel * C).*(y - kernel * C)) + lambda * C' * kernel * C)
            cvx_end
            kernelClassifierObj = model.classify.KernelClassifier(X, y, kernel, C);
        end
    end
end
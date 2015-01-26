classdef SupportVectorMachine < handle
    properties
        X
        y
        sigma
        alpha
        K
    end
    
    methods
        function supportVectorMachineObj = SupportVectorMachine(X, y, sigma, alpha, K)
            supportVectorMachineObj.X = X;
            supportVectorMachineObj.y = y;
            supportVectorMachineObj.sigma = sigma;
            supportVectorMachineObj.alpha = alpha;
            supportVectorMachineObj.K = K;
        end
        
        function labels = predict(obj, data)
            
            sig = obj.sigma;
            
            [n, ~] = size(obj.X);
            [m, d] = size(data);
            points = zeros(m+n, d);
            
            points(1:n,:) = obj.X;
            points(n+1:m+n,:) = data;
            
            kernel = exp(-1*(squareform(pdist(points)).^2)./(2*(sig^2)));
            kernel = kernel(n+1:n+m, 1:n);
            
            labels = sign(kernel * obj.alpha);
        end
    end
        
    methods (Static)
        function supportVectorMachineObj = train(X, y, options)
                
            sig = options.sigma;
            lambda = options.lambda;
                
            [n, ~] = size(X);
                
            Y = diag(y);
            kernel = exp(-1*(squareform(pdist(X)).^2)./(2*(sig^2)));
            Q = Y * (kernel ./ (2*lambda)) * Y;
                
            cvx_begin
                variable b(n, 1)
                maximize (sum(b) - 0.5 * b' * Q * b)
                subject to
                    y' * b == 0;
                    zeros(n, 1) <= b <= repmat(1/n, n, 1);
            cvx_end
                
            a = Y * b ./ (2 * lambda);
                
            supportVectorMachineObj = model.classify.SupportVectorMachine(X, y, sig, a, kernel);
                
        end
    end
end
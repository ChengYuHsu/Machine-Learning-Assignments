classdef LaplacianSupportVectorMachine < handle
    properties
    	X
        y
        a
        K
        L
        r
    end
    
    methods
    	function laplacianSupportVectorMachineObj = LaplacianSupportVectorMachine(X, y, a, K, L, r)
    		laplacianSupportVectorMachineObj.X = X;
            laplacianSupportVectorMachineObj.y = y;
            laplacianSupportVectorMachineObj.a = a;
            laplacianSupportVectorMachineObj.K = K;
            laplacianSupportVectorMachineObj.L = L;
            laplacianSupportVectorMachineObj.r = r;
        end
        
        function labels = predict(obj, data)
            gam = obj.r;
            
            [n, ~] = size(obj.X);
            [m, d] = size(data);
            points = zeros(m+n, d);
            
            points(1:n,:) = obj.X;
            points(n+1:m+n,:) = data;
            
            kernel = exp(-1*gam*(squareform(pdist(points)).^2));
            kernel = kernel(n+1:n+m, 1:n);
            labels = sign(kernel * obj.a);
        end
        
	end

    methods(Static)
        
        
        function laplacianSupportVectorMachineObj = train(X, y, opts)
        	
            gam = opts.gamma;
            lambda = opts.lambda;
            miu = opts.miu;
            
            [n, d] = size(X);
            
            labeledX = X(y~=0, :);
            labeledY = y(y~=0);
            
            [l, ~] = size(labeledX);
            
            newX = zeros(n, d);
            newX(1:l, :) = labeledX;
            newX(l+1:n, :) = X(y==0, :);
            
            newY = zeros(n, 1);
            newY(1:l) = labeledY;
            
            
            % solve dual problem
            Y = diag(labeledY);
            
            J = zeros(l, n);
            J(1:l, 1:l) = eye(l);
            
            % kernel matrix
            kernel = exp(-1* gam *(squareform(pdist(X)).^2));
            
            % build laplacian

			S = exp( -1 * gam * (squareform(pdist(X)).^2) );
            D = diag(sum(S, 2));
			laplacian = D - S;
            
            Q = Y*J*kernel*inv(2*lambda*eye(n)+2*(miu/n^2)*laplacian*kernel)*J'*Y;
            
            cvx_begin quiet
                variable f(l, 1)
                maximize (sum(f) - 0.5 * f' * Q * f)
                subject to
                    labeledY' * f == 0;
                    zeros(l, 1) <= f <= repmat(1/l, l, 1);
            cvx_end
                		
            alpha = inv(2*lambda*eye(n)+2*(miu/n^2)*laplacian*kernel)*J'*Y*f;
            
    		laplacianSupportVectorMachineObj = model.classify.LaplacianSupportVectorMachine(newX, newY, alpha, kernel, laplacian, gam);
        end

        

    end
end
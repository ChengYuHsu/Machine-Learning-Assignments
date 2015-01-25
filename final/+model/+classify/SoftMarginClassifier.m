%%% the dummy classifier predicts the label by randomly assigning a label
%%% based on the ratio of +1/-1 examples in training data

classdef SoftMarginClassifier < handle
   properties
      posRatio; % the ratio of examples with label +1
      w;
      b;
      e;
   end
   
   methods
       function softMarginClassifierObj = SoftMarginClassifier (pRatio)  % constructor
           softMarginClassifierObj.posRatio = pRatio;
       end
       function predictedLabel = predict (obj, X)
          tempMatrix = obj.w'*X' - obj.b;
          
          predictedLabel = sign(tempMatrix)';
           
       end
       
   end
   
   methods (Static)
      function softMarginClassifierObj = train (X, y)
        softMarginClassifierObj = model.classify.SoftMarginClassifier(length(X(y==1,:))/length(X));
        
        [n, m] = size(X);
        
        G = zeros(2*n, m+n+1);
        G = double(G);
        
        negativeY = repmat(y, 1, m).*-1;
        G(1:n,1:m) = X(1:n, 1:m).*negativeY;
        
        G(1:n, m+1) = y(1:n, 1);
        
        identityMatrixN = eye(n);
        
        G(1:n, m+2:m+n+1) = -1.*identityMatrixN;
        G(n+1:2*n, m+2:m+n+1) = -1.*identityMatrixN;
        
        Q = zeros(m+n+1, m+n+1);
        identityMatrixM = eye(m);
        Q(1:m,1:m) = identityMatrixM;
        
        c = zeros(m+n+1, 1);
        c(m+2:m+n+1, 1) = 1;
        
        h = zeros(2*n, 1);
        h(1:n, 1) = -1;
        
        cvx_begin
            variable x(m+n+1, 1)
            minimize (x'*Q*x + c'*x)
            subject to
                G*x <= h;
        cvx_end
        
        softMarginClassifierObj.w = x(1:m, 1);
        softMarginClassifierObj.b = x(m+1, 1);
        softMarginClassifierObj.e = x(m+2:m+n+1, 1);
        
      end
   end
end
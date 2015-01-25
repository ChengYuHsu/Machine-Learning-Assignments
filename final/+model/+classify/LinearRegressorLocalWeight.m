
classdef LinearRegressorLocalWeight < model.regressor.LinearRegressor
   properties
        X
        y
        tau
   end

   methods
       function linearRegressorLocalWeightObj = LinearRegressorLocalWeight(X, y)  % constructor
            linearRegressorLocalWeightObj@model.regressor.LinearRegressor();
            linearRegressorLocalWeightObj.X = X;
            linearRegressorLocalWeightObj.y = y;
            linearRegressorLocalWeightObj.w = zeros(1);
       end
       function predictedValue = predict (obj, newX, cfg)
            obj.tau = cfg('tau');

            [N, d] = size(obj.X);
            [nN, nd] = size(newX);
            X =  [ones(N, 1), obj.X];
            r = obj.y;
            predictedValue = zeros(nN, 1);
            for j=1: nN
                L = zeros(N, N);

                for i=1 : N
                    L(i, i) = obj.localWeight(newX(j), obj.X(i));
                end

                cvx_begin
                    variable w(d+1)
                    minimize( (X * w - r )' * L * (X * w - r) )
                cvx_end
                w

                predictedValue(j) = [1, newX(j)]*w*L(j, j);

            end

       end

       function weight = localWeight (obj, new_x, X)
            weight = 0.0;

            a_norm = norm(new_x - X)^2;
            tau = 2*(obj.tau^2);
            weight = weight + exp(-1.0*(a_norm/tau));

       end

   end

   methods (Static)
      function linearRegressorLocalWeightObj = train (X, y)
        linearRegressorLocalWeightObj = model.regressor.LinearRegressorLocalWeight(X, y);
      end
   end
end

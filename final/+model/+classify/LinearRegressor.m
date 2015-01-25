classdef LinearRegressor < handle
   properties
      w; % parameters
   end

   methods
       function linearRegressorObj = LinearRegressor ()  % constructor

       end
       function predictedValue = predict (obj, X)
            [N, d] = size(X);
            data = ones(N, d+1);
            data(1:N, 2:d+1) = X;
            w = obj.w;
            predictedValue = data*w;
            w

       end

       function empiricalError = emp(obj, X, y, w)
            empiricalError = 0.0;
            for i = 1:length(X)
                x_data = [1, X(i)]';
                empiricalError = empiricalError + (y(i) - w'*x_data)^2;
            end
       end

       function trained_w = gradient_descent(obj, X, y)

            [N, d] = size(X);
            trained_w = eye(d+1, 1);
            obj.w = eye(d+1, 1);
            learning_rate = 0.0001;
            prev_emp = inf;
            interation = 0;
            while true
                for i=1:N
                    trained_w = trained_w + 2 * learning_rate * (y(i) - trained_w' * [1, X(i)]' ) * [1, X(i)]';
                end
                emp = obj.emp(X, y, trained_w);
                if abs(prev_emp - emp)<=0.001
                    break;
                else
                    prev_emp = emp;
                end
                interation = interation + 1;
            end
            interation
            prev_emp
       end

   end

   methods (Static)
      function linearRegressorObj = train (X, y)
        linearRegressorObj = model.regressor.LinearRegressor();
        linearRegressorObj.w = linearRegressorObj.gradient_descent(X, y);
      end
   end
end

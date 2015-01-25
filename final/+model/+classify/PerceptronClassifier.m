%%% the dummy regressor predicts the value by mean value of y

classdef PerceptronClassifier < handle
   properties
      w;
      b;
   end

   methods
        function perceptronClassifierObj = PerceptronClassifier(w, b)  % constructor
           
        end
        
        function predictedValue = predict (obj, X)
           predictedValue = sign(X * obj.w + obj.b);
           w = obj.w
           b = obj.b
           
        end
       
        function iterate(obj, X, y)
            [N, d] = size(X);
            % random initialization
            obj.w = rand([d, 1]);
            obj.b = rand();
            
            max_iter = 5*10^5;
            iter = 1;
            step_size = 10^-7;
            deduction_rate = 1-10^-6;
            converge_criteria = 10^-7;
            last_emp = inf;
            while true
                emp = 0;
                w_delta_value = 0;
                label_val = obj.label(X);
                new_x = [X, ones(N, 1)];
                
                w_delta_value = w_delta_value + new_x'*(y - label_val);
                %b_delta_value = b_delta_value + (y - label_val);
                
                w_delta_value = step_size * w_delta_value;
                %b_delta_value = step_size * b_delta_value;
                
                new_w = obj.w + w_delta_value(1:d);
                new_b = obj.b + w_delta_value(d+1);
                
                
                norm(new_w-obj.w);
                % norm(new_w-obj.w)
                if norm(w_delta_value) > converge_criteria
                    obj.w = new_w;
                    obj.b = new_b;
                    iter = iter + 1;
                     step_size = 10^-7 * (1.0-iter/max_iter);
                    if iter > max_iter
                        break;
                    end
                else
                    break;
                    
                end
            end
            final_iter = iter
        end
        
        function labelValue = label(obj, X)
            labelValue = sign(X * obj.w + obj.b);
        end
        
        
   end

   methods (Static)
       
      function perceptronClassifierObj = train (X, y)
        perceptronClassifierObj = model.classifier.PerceptronClassifier();
        perceptronClassifierObj.iterate(X, y);
      end
      
   end
end

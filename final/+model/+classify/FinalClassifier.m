classdef FinalClassifier < handle
    properties
        X
        y
        svm
    end
    
    methods
        
        function finalClassifier = FinalClassifier(X, y)
            addpath('tools/libsvm/matlab/');
            finalClassifier.X = X;
            finalClassifier.y = y;
            finalClassifier.svm = svmtrain(y(y~=0), X(y~=0));
        end

        function predictedLabels = predict (obj, X)
            [n, ~] = size(X);
            [predictedLabels, accuracy, ~] = svmpredict(ones(n, 1), X, obj.svm);
            accuracy
        end
   end

    methods (Static)

        function finalClassifier = train (X, y)
            finalClassifier = model.classify.FinalClassifier(X, y);
        end

    end

    

end
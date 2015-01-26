classdef FinalClassifier < handle
    properties
        X
        y
        w_lda
        lapSVM
        svm
    end
    
    methods
        
        function finalClassifier = FinalClassifier(X, y, w_lda, lapSVM, svm)
            finalClassifier.X = X;
            finalClassifier.y = y;
            finalClassifier.w_lda = w_lda;
            finalClassifier.lapSVM = lapSVM;
            finalClassifier.svm = svm;
        end

        function predictedLabels = predict (obj, data)
            [~, d] = size(data);
            
            processed = zscore(data);
            processed = processed * obj.w_lda(:,1:d)';
            predictedLabels = obj.svm.predict(processed);
        end
   end

    methods (Static)

        function finalClassifier = train (X, y)

            % normalize data
            X = zscore(X);

            % propagate labels
            lapopt.gamma = 19;
            lapopt.lambda = 1.3;
            lapopt.miu = 1.4;

            lap_SVM = model.classify.LaplacianSupportVectorMachine.train(X, y, lapopt);
            propagated = lap_SVM.predict(X);

            % perform dimension reduction
            [~, coeffs] = model.dimension.LDA.project(X, propagated);

            % filter out insignificant factors
            threshold = 0.4;
            coeffs = ge(coeffs, threshold) .* coeffs + le(coeffs, -1*threshold) .* coeffs;
            
            
            % reduce to 2D
            [n, d] = size(X);
            X = X * coeffs(:,1:d)';

            % train a support vector machine
            svmopt.gamma = 20;
            svmopt.lambda = 1.31;
            svmopt.miu = 0;

            % 8-fold cross validation
            folds = 8;
            idx = crossvalind('Kfold', propagated, folds);
            err = 0;
            min_arr = inf;

            for i=1:folds
                test = (idx == i);
                train = ~test;
                
                candidate_svm = model.classify.LaplacianSupportVectorMachine.train(X(train,:), propagated(train,:), svmopt);
                labels = candidate_svm.predict(X(test,:));

                testing_err = (nnz(propagated(test) - labels) / length(propagated(test)));
                
                if testing_err < min_arr
                    n_svm = candidate_svm;
                    min_arr = testing_err;
                end

            end

            

            

            finalClassifier = model.classify.FinalClassifier(X, y, coeffs, lap_SVM, n_svm);
        end

    end

    

end
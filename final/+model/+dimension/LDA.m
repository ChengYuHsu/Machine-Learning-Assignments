classdef LDA < handle
    methods (Static)
        function [classes, w] = project(X, y)
            
            % Determine size of input data
            [n, d] = size(X);


            classes = unique(y);
            labels = length(classes);
            


            groups = zeros(labels, 1);
            means = zeros(labels, d);
            covar = zeros(d, d);
            w = zeros(labels, d+1);

            

% Loop over classes to perform intermediate calculations
            for i = 1:labels
    % Establish location and size of each class
                group      = (y == classes(i));
                groups(i)  = sum(double(group));
    
        % Calculate group mean vectors
                means(i,:) = mean(X(group,:));
    
    % Accumulate pooled covariance information
                covar = covar + ((groups(i) - 1) / (n - labels) ) .* cov(X(group,:));
            end






        % Use the sample probabilities
            PriorProb = groups / n;
    

% Loop over classes to calculate linear discriminant coefficients
            for i = 1:labels
    % Intermediate calculation for efficiency
    % This replaces:  GroupMean(g,:) * inv(PooledCov)
                temp = means(i,:) / covar;
    
    % Constant
                w(i,1) = -0.5 * temp * means(i,:)' + log(PriorProb(i));
    
    % Linear
                w(i,2:end) = temp;
            end
        end
    end
    
end
    
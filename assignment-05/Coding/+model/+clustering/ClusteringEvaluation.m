classdef ClusteringEvaluation

	methods(Static)

		function quality = evaluate(data, indicators, k)
			
			% centroids
			[~, n] = size(data);
			centroids = zeros (k, n);
			for i=1:k
				for j=1:n
					centroids(i, j) = mean(data(indicators(:,i) == 1, j));
				end
			end

			% inter-cluster separation
			interClusterSeparation = sum(pdist(centroids) .^ 2);

			% intra-cluster separation
			intraClusterSeparation = 0;
			for i=1:k
				[g, ~] = size(data(indicators(:,i)==1));
				points = zeros(g+1, n);
				points(1,:) = centroids(i,:);
				points(2:g+1,:) = data(indicators(:,i)==1,:);

				distMat = squareform(pdist(points)) .^ 2;
				
				intraClusterSeparation = intraClusterSeparation + (sum(distMat(1, 2:g+1)) / g);
			end
			
			quality = interClusterSeparation / intraClusterSeparation;


		end

		

	end

end
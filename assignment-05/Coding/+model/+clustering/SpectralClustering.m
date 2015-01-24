classdef SpectralClustering < model.clustering.KmeansClustering
	
	properties
		cfg
	end

	methods (Static)

		function clusterIndicators = cluster(X, k, cfg)

        	obj = model.clustering.SpectralClustering(X, k, cfg);

        	obj.spectralCluster();

        	clusterIndicators = obj.indicators;

      	end	

   end

	methods
		function objInstance = SpectralClustering(X, k, cfg)
			objInstance@model.clustering.KmeansClustering(X, k);
			objInstance.X = X;
			objInstance.k = k;
			objInstance.cfg = cfg;
		end

		function spectralCluster(obj)

			[m, ~] = size(obj.X);

			[l, d, s] = obj.buildLaplacian();
			
			opt = struct('issym', true, 'isreal', true);
			[v, ~] = eigs(l, d, obj.k, 'sm', opt);
			
			obj.X = v;

			obj.indicators = model.clustering.KmeansClustering.cluster(obj.X, obj.k);

		end

		function [l, d, s] = buildLaplacian(obj)

			sim = obj.cfg('similarity');

			[m, ~] = size(obj.X);

			s = zeros(m, m);

			if (strcmp(sim, 'eNN'))
			
				s = obj.buildENNSimilarityMatrix();

			elseif (strcmp(sim, 'eBall'))
			
				s = obj.buildEBallSimilarityMatrix();

			elseif (strcmp(sim, 'Gaussian'))
			
				s = obj.buildGaussianSimilarityMatrix();
			
			end

			d = diag(sum(s, 2));
			
			l = d - s;
				
		end

		function s = buildENNSimilarityMatrix(obj)

			e = obj.cfg('e');

			[m, ~] = size(obj.X);

			s = zeros(m, m);
			
			for i=1:m

				idx = knnsearch(obj.X, obj.X(i,:), 'k', e+1);

				for j=1:e+1

					if idx(j) ~= i

						dist = pdist2(obj.X(i,:), obj.X(idx(j),:));

						s(i, idx(j)) = 1/dist;

						s(idx(j), i) = 1/dist;

					end

				end

			end

		end

		function s = buildEBallSimilarityMatrix(obj)
			
			e = obj.cfg('e');
			
			[m, ~] = size(obj.X);

			s = zeros(m, m);

			distMat = squareform(pdist(obj.X));
			

			s(distMat < e) = distMat(distMat < e);
			
			s(s ~= 0) = 1 ./ s(s ~= 0);
			
		end

		function s = buildGaussianSimilarityMatrix(obj)
			
			sigma = obj.cfg('sigma');

			[m, ~] = size(obj.X);
			
			s = zeros(m, m);

			s = exp( -1 * (squareform(pdist(obj.X)).^2) ./ (sigma ^ 2) );

			s = s - diag(ones(m, 1));

		end

	end

	

end
classdef SpectralClustering < model.clustering.KmeansClustering

	properties
		cfg
	end
	methods
		function objInstance = SpectralClustering(X, k, cfg)
			objInstance@model.clustering.KmeansClustering(X, k);
			objInstance.X = X;
			objInstance.k = k;
			objInstance.cfg = cfg;
		end

		function spectralCluster(obj)
			[m, n] = size(obj.X);

			[l, d, s] = obj.buildLaplacian();

			[v, dummy] = eigs(l);
			
			u = zeros(m, obj.k);

			for i=1:obj.k
				u(:,i) = v(:,i);
			end
			
			obj.X = u;

			obj.indicators = model.clustering.AnotherKMeansClustering.cluster(obj.X, obj.k);

			

		end

		function [l, d, s] = buildLaplacian(obj)
			sim = obj.cfg('similarity');
			[m, n] = size(obj.X);
			s = zeros(m, m);

			if (strcmp(sim, 'eNN'))
				s = obj.buildENNLaplacian();
			elseif (strcmp(sim, 'eBall'))
				s = obj.buildEBallLaplacian();
			elseif (strcmp(sim, 'Gaussian'))
				s = obj.buildGaussianLaplacian();
			end

			d = zeros(m, m);
			for i=1:m
				d(i, i) = sum(s(i,:));
			end 
			
			l = d - s;
				
		end

		function s = buildENNLaplacian(obj)
			e = obj.cfg('e');
			[m, n] = size(obj.X);
			s = zeros(m, m);

			for i=1:m
				idx = knnsearch(obj.X, obj.X(i,:), 'k', e+1);
				for j=1:e+1
					if idx(j) ~= i
						dist = pdist2(obj.X(i,:), obj.X(j,:))+10^-10;
						s(i, j) = 1.0/dist;
						s(j, i) = 1.0/dist;
					end
				end
			end

		end

		function s = buildEBallLaplacian(obj)
			e = obj.cfg('e');
			[m, n] = size(obj.X);
			s = zeros(m, m);

			distMat = squareform(pdist(obj.X))+10^-20;
			distMat = 1.0./distMat;

			s(distMat < e) = distMat(distMat < e);

		end

		function s = buildGaussianLaplacian(obj)
			sigma = obj.cfg('sigma');
			[m, n] = size(obj.X);
			s = zeros(m, m);

			distMat = squareform(pdist(obj.X));
			distMat = -1*distMat.^2;
			distMat = distMat./(sigma^2);
			distMat = exp(distMat);

			s = distMat;
		end

	end

	methods (Static)
		function clusterIndicators = cluster(X, k, cfg)
        	obj = model.clustering.SpectralClustering(X, k, cfg);
        	obj.spectralCluster();
        	clusterIndicators = obj.indicators;
      	end	
   end

end
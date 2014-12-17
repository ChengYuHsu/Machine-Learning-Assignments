classdef KmeansClustering < handle
	
	properties
		X
		k
		seeds
		indicators
	end

	methods
		function objInstance = KmeansClustering(X, k)
			%objInstance = KmeansClustering();
			objInstance.X = X;
			objInstance.k = k;
		end


		function indicators = KmeansCluster(obj)

      		[m, n] = size(obj.X);
      		labels = zeros(m, obj.k);

      		seeds = obj.KmeansPlusPlusInit;

      		while(1)
      			labels = zeros(m, obj.k);
      			for i=1:m
      				dist = inf;
      				label = 0;
      				instance = obj.X(i,:);
      				
      				for j=1:obj.k
      					seed = seeds(j,:);
      					nDist = pdist2(seed, instance);
      					
      					if dist > nDist
      						label = j;
      						dist = nDist;
      					end
      				end
      				labels(i, label) = 1;
      			end

      			newSeeds = zeros(obj.k, n);
      			for i=1:obj.k
      				group = labels(:,i);
      				groupSize = 0;
      				for j=1:m
      					if group(j) == 1
      						newSeeds(i,:) = newSeeds(i,:) + obj.X(j,:);
      						groupSize = groupSize + 1;
      					end
      				end
      				newSeeds(i,:) = newSeeds(i,:)./groupSize;
      			end

      			if isequal(newSeeds, seeds)
      				break;
      			end
      			seeds = newSeeds;
      		end
      		
      		obj.indicators = labels;

      	end

      	function seeds = KmeansPlusPlusInit(obj)
      		[m, n] = size(obj.X);
      		seeds = zeros(obj.k, n);
      		
      		seedSet = zeros(obj.k, 1);

      		seedSet(1) = randi(m);
      		seeds(1,:) = obj.X(seedSet(1),:);
      		numSeeds = 1;

      		while(1)
      			farest = 0;
      			newSeed = 0;
      			base = obj.X(seedSet(numSeeds),:);
      			for i = 1:m
      				if sum(ismember(seedSet, i)) == 0 
      					dist = pdist2(base, obj.X(i,:));
      					if dist > farest
      						newSeed = i;
      						farest = dist;
      					end
      				end
      			end

      			numSeeds = numSeeds + 1;
      			seedSet(numSeeds) = newSeed;
      			seeds(numSeeds,:) = obj.X(newSeed,:);
      			if numSeeds == obj.k
      				break;
      			end
      		end

      	end

	end


	methods (Static)
		function clusterIndicators = cluster(X, k)
        	obj = model.clustering.KmeansClustering(X, k);
        	obj.KmeansCluster();
        	clusterIndicators = obj.indicators;

      	end

      	
   end
end

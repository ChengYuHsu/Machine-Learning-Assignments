import model.clustering.KmeansClustering
import model.clustering.SpectralClustering

clear;

data = importdata('./data/data.txt');

% epsilon nearest neighbor
%indicators = SpectralClustering.cluster(data, 2, containers.Map({'similarity', 'e'}, {'eNN', 5}));

% epsilon ball
indicators = SpectralClustering.cluster(data, 2, containers.Map({'similarity', 'e'}, {'eBall', 1.3}));

% gaussian
%indicators = SpectralClustering.cluster(data, 2, containers.Map({'similarity', 'sigma'}, {'Gaussian', 0.5}));

% k means
%indicators = KmeansClustering.cluster(data, 2);


hold on;
scatter(data(indicators(:,1) == 1, 1), data(indicators(:,1) == 1, 2), 'r');
scatter(data(indicators(:,2) == 1, 1), data(indicators(:,2) == 1, 2), 'b');
hold off;

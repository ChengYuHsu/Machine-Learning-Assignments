import model.clustering.KmeansClustering
import model.clustering.SpectralClustering
import model.clustering.ClusteringEvaluation

clear;

data = importdata('./data/data.txt');

% epsilon nearest neighbor
% indicators = SpectralClustering.cluster(data, 2, containers.Map({'similarity', 'e'}, {'eNN', 3}));

% epsilon ball
% indicators = SpectralClustering.cluster(data, 2, containers.Map({'similarity', 'e'}, {'eBall', 1.2}));

% gaussian
% indicators = SpectralClustering.cluster(data, 2, containers.Map({'similarity', 'sigma'}, {'Gaussian', 0.2}));

% k means
indicators = KmeansClustering.cluster(data, 2);

quality = ClusteringEvaluation.evaluate(data, indicators, 2);
quality

hold on;
scatter(data(indicators(:,1) == 1, 1), data(indicators(:,1) == 1, 2), 'r');
scatter(data(indicators(:,2) == 1, 1), data(indicators(:,2) == 1, 2), 'b');
hold off;

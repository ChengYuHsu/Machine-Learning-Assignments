import model.clustering.KmeansClustering
import model.clustering.SpectralClustering

clear;

data = importdata('./data/data.txt');

 indicators = SpectralClustering.cluster(data, 2, containers.Map({'similarity', 'e'}, {'eBall', 1.2}));

%indicators = KmeansClustering.cluster(data, 2);


hold on;
scatter(data(indicators(:,1)==1,1), data(indicators(:,1)==1,2), 'r');
scatter(data(indicators(:,2)==1,1), data(indicators(:,2)==1,2), 'b');
%scatter(data(indicators(:,3)==1,1), data(indicators(:,3)==1,2), 'g');
hold off;

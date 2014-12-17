
classdef AnotherKMeansClustering < handle
   properties
      
   end
         
   methods (Static)
      function Y = cluster (X, k)
          [m, n] = size(X);
          means=zeros(k, n);
          newMeans = zeros(k, n);
          
          %%% random guess %%%
          %index = randperm(m);
          %for i=1:k
          %  means(i, :) = X(index(i), :)
          %end  
          
           %-- Kmeans ++ initialization --%
          index = randperm(m);
          means(1, :) = X(index(1), :);
          %-- create d matrix --%
          for knum=2:k
              D = zeros(m, n);
              for i=1:m
                  D(i, :) = norm(means(1, :) - X(i,:), 2);
                  for j=2:knum
                      dis = norm(means(j, :) - X(i,:), 2);
                      if D(i, :) > dis
                         D(i, :) = dis; 
                      end    
                  end
              end
              [maxV, pos] = max( norm(D,2) );
              means(knum, :) = X(pos, :);
          end
          
          %-- loop --%
          stop = 10^-4;
          while true
              meansCount = zeros(k,n);
              numCount = zeros(k, 1);
              indicator = zeros(m, 1);
              %-- scan all instance --%
              for i=1:m
                  %-- determin cluster for a instance i --%
                  min = norm(means(1, :) - X(i, :) , 2);
                  minpos = 1;
                  for j=2:k
                     tmp = norm(means(j, :) - X(i, :) , 2);
                     if tmp < min
                         min = tmp;
                         minpos = j;
                     end
                  end
                  meansCount (minpos, :) =  meansCount (minpos, :) + X(i, :);
                  numCount (minpos, 1) = numCount (minpos, 1)+1;
                  indicator (i, 1) = minpos;
              end
              %-- update means --%  
              change = 0;
              for i=1:k
                  newMeans(i, :) = meansCount(i, :) / numCount(i, 1);
                  change = change + norm(newMeans(i,:) - means(i,:), 2);
                  means(i, :) = newMeans(i, :);
              end    
              if change < stop
                  break;
              end    
          end
          %-- construct Y by indicator--%
          Ans = zeros(m, k);
          for i=1:m
              Ans (i, indicator(i)) = 1;
          end
          Y = Ans;
      end
   end
end
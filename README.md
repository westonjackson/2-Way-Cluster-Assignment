#2-Way Clustering Assignment Simulator

####This code randomly simulates a 2D data set with 2-Way Cluster Assignment.

Data points: xi...xn

Clusters: yj...yn


####Cluster Parameters:

2-Way Assignment Probability: (j < j') -> (pi)jj'

Cluster Mean: (mu)jj'

Cluster Covariance: (Sigma)jj'


####Algorithm

Given parameters mu(1...k), Sigma(1...k), pi(1,1..k,k). All clusters will have centers at mu(1...k) with standard deviations Sigma(1...k).

For a given point xi:

1. Assign xi to 2-Way cluster jj' with probability pi(jj').
2. 
2. Assign probability ui = Uniform(0,1) to xi, which creates the following normal distribution for xi:
3. 
N(xi; ui muj + (1 - ui)muj', ui Sigmaj + (1 - ui)Sigmaj'

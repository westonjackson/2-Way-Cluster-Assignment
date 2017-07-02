# 2-Way Clustering Assignment

### files
- TwoWaykmeans.py: Contains executable for R16S real data and functions for running two-way k-means. Run "python TwoWaykmeans" with OTU data present in order to see results. Import this file to use TwoWayCluster function.
- test.py: Test data for two-way k-means algorithm. Run "python test.py" to see sample run.

## TwoWaykmeans.TwoWayCluster(X,k,num_rounds)
### params
- X: numpy data matrix
- k: number of clusters
- num_rounds: number of rounds

### attributes
- C: k cluster centers as columns
- phi: 2-sparse vector cluster assignments (u, 1 - u) for each x_i



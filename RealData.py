import pandas as pd
import numpy as np
import TwoWayEM
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':
    sample_names = pd.read_table("combined.clean.an.0.03.subsample.shared", header=0, delimiter="\t")
    sample_names.drop(sample_names.columns[[0,1,2]], axis=1, inplace=True)
    sample_names = sample_names.as_matrix().astype(float)[:,0:20]

    data = pd.read_table("combined.clean.an.thetayc.0.03.lt.ave.pcoa.axes", header=0, delimiter="\t")
    labels = data["group"]
    data.drop(data.columns[[0]], axis=1, inplace=True)
    X = data.as_matrix().astype(float)
    X = X[:,0:3]
    
    print len(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.T[0], X.T[1], X.T[2])
    plt.show()
    
    
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:,1]
    indices = np.where(distances < .02)
    X = X[indices]
    sample_names = sample_names[indices]

    print len(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.T[0], X.T[1], X.T[2])
    plt.show()
    
    k = 5
    results = TwoWayEM.classify(X, k, cov_init_magnitude=.001, num_rounds=1, two_way_significance_level = .99)
    
    w = results[0]
    u = results[1]
    pi = results[2]
    mu = results[3]
    sigma = results[4]
    
    c1 = np.max(w, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.T[0], X.T[1], X.T[2], c=c1)
    ax.scatter(mu.T[0], mu.T[1], mu.T[2], c='r')
    plt.show()
    
    results = TwoWayEM.classify(X, k, cov_init_magnitude=.001, num_rounds=5, two_way_significance_level = .99)
    
    w = results[0]
    u = results[1]
    pi = results[2]
    mu = results[3]
    sigma = results[4]
    
    c1 = np.max(w, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.T[0], X.T[1], X.T[2], c=c1)
    ax.scatter(mu.T[0], mu.T[1], mu.T[2], c='r')
    plt.show()

    c2 = np.argmax(w, axis=1)
    count = np.zeros(((int(k*k)),20))
    for i in range(int(k*k)):
        for j in range(len(X)):
            otus = np.argsort(sample_names[j])[-2:][::-1]
            assignment = c2[j]
            cluster_1 = assignment/len(mu)
            cluster_2 = assignment%len(mu)
            if cluster_1*len(mu) + cluster_2 == i or cluster_2*len(mu) + cluster_1 == i:
                count[i][otus[0]] = count[i][otus[0]] + 1
                count[i][otus[1]] = count[i][otus[1]] + 1

    count = count.T
    for i in range(len(count)):
        for j in range(len(count[0]) - 1):
            cluster_1 = j/len(mu)
            cluster_2 = j%len(mu)
            if cluster_1 <= cluster_2:
                print '{} & '.format(int(count[i][j])),
        print '{} \\\\ \hline '.format(int(count[i][len(count[0]) - 1]))


    count = np.zeros((int(k*k),20))
    for i in range(int(k*k)):
        for j in range(len(X)):
            otus = np.argmax(sample_names[j])
            assignment = c2[j]
            cluster_1 = assignment/len(mu)
            cluster_2 = assignment%len(mu)
            if cluster_1*len(mu) + cluster_2 == i or cluster_2*len(mu) + cluster_1 == i:
                count[i][otus] = count[i][otus] + 1

    count = count.T
    for i in range(len(count)):
        for j in range(len(count[0]) - 1):
            cluster_1 = j/len(mu)
            cluster_2 = j%len(mu)
            if cluster_1 <= cluster_2:
                print '{} & '.format(int(count[i][j])),
        print '{} \\\\ \hline '.format(int(count[i][len(count[0]) - 1]))




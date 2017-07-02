import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.cluster.vq import kmeans
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
#from sklearn import linear_model
#from sklearn.linear_model import LassoLars
#from sklearn.linear_model import LinearRegression

def minimum(p, mu_1, mu_2):
    return np.dot((mu_2 - mu_1).T,(mu_2 - p))*(1/(float(np.linalg.norm(mu_2 - mu_1)**2)))

#def get_lasso(p,C):
#    lasso = linear_model.Lars(n_nonzero_coefs = 2, positive = True)
#    least_squares = lasso.fit(C,p).coef_
#    indices = least_squares.argsort()[-2:][::-1]
#    print indices
#    uijj = minimum(p, C.T[indices[0]],C.T[indices[1]])
#    if uijj > 1:
#        uijj = 1
#    if uijj < 0:
#        uijj = 0
#    phi = np.zeros((1,5))
#    phi[0,indices[0]] = uijj
#    phi[0,indices[1]] = 1 - uijj
#    print "result: ", phi
#    return lasso.fit(C,p)


def zero_one_lasso(X, C):
    error = 0
    summation = 0
    phi = np.zeros((len(C[0]), len(X)))
    for i in range(len(X)):
        p = X[i]
        mindistance = np.inf
        index_1 = 0
        index_2 = 0

        for j in range(len(C[0])):
            for k in range(j + 1, len(C[0])):
                u_jk = minimum(p, C.T[j],C.T[k])
                if u_jk > 1:
                    u_jk = 1
                if u_jk < 0:
                    u_jk = 0
                dist = np.linalg.norm(p + u_jk*(C.T[k] - C.T[j]) - C.T[k])
                if dist < mindistance:
                    index_1 = j
                    index_2 = k
                    u = u_jk
                    mindistance = dist
        phi[index_1][i] = u
        phi[index_2][i] = 1 - u
        error = error + mindistance
    print "Objective: ", error
    return phi

def TwoWayCluster(X, k, num_rounds):
    n = len(X)
    d = len(X[0])

    mu = kmeans(X, k)[0]
    mu = mu[mu[:,0].argsort()]

    phi = np.zeros((k, n))
    C = mu.T

    for i in range(num_rounds):
        phi = zero_one_lasso(X, C)
        C = np.dot(np.dot(np.linalg.inv(np.dot(phi,phi.T)),phi), X).T

    return C,phi

if __name__ == '__main__':
    sample_names = pd.read_table("otudata/combined.clean.an.0.03.subsample.shared", header=0, delimiter="\t")
    sample_names.drop(sample_names.columns[[0,1,2]], axis=1, inplace=True)
    sample_names = sample_names.as_matrix().astype(float)[:,0:20]

    data = pd.read_table("otudata/combined.clean.an.thetayc.0.03.lt.ave.pcoa.axes", header=0, delimiter="\t")
    labels = data["group"]
    data.drop(data.columns[[0]], axis=1, inplace=True)
    points = data.as_matrix().astype(float)
    points = points[:,0:3]


    n = len(points)
    d = len(points[0])

    k = 5
    mu = kmeans(points, k)[0]

    phi = np.zeros((k, n))
    C = mu.T

    c1 = np.max(phi.T, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points.T[0], points.T[1], points.T[2], c=c1)
    ax.scatter(C[0], C[1], C[2], c='r')
    plt.show()


    for i in range(5):
        phi = zero_one_lasso(points, C)
        C = np.dot(np.dot(np.linalg.inv(np.dot(phi,phi.T)),phi),points).T
    c1 = np.max(phi.T, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points.T[0], points.T[1], points.T[2], c=c1)
    ax.scatter(C[0], C[1], C[2], c='r')
    plt.show()


    y = np.zeros((len(points),2))

    for i in range(len(points)):
        indices = np.nonzero(phi.T[i])
        y[i] = indices[0]

    count = np.zeros((k,20))
    for i in range(len(points)):
        otus = np.argmax(sample_names[i])
        assignment = np.argmax(phi.T[i])
        count[assignment][otus] = count[assignment][otus] + 1

    count = count.T
    for i in range(len(count)):
        for j in range(len(count[0]) - 1):
            print '{} & '.format(int(count[i][j])),
        print '{} \\\\ \hline '.format(int(count[i][len(count[0]) - 1]))


    count = np.zeros((int(k*k),20))
    for i in range(len(points)):
        otus = np.argmax(sample_names[i])
        #otus = np.argsort(sample_names[i])[-2:][::-1]
        assignment = phi.T[i].argsort()[-2:][::-1]
        cluster_1 = assignment[0]
        cluster_2 = assignment[1]
        count[cluster_1*len(mu) + cluster_2][otus] = count[cluster_1*len(mu) + cluster_2][otus] + 1
        count[cluster_2*len(mu) + cluster_1][otus] = count[cluster_2*len(mu) + cluster_1][otus] + 1


    count = count.T
    for i in range(len(count)):
        for j in range(len(count[0]) - 1):
            cluster_1 = j/len(mu)
            cluster_2 = j%len(mu)
            if cluster_1 < cluster_2:
                print '{} & '.format(int(count[i][j])),
        print '{} \\\\ \hline '.format(int(count[i][len(count[0]) - 1]))

    count = np.zeros((int(k*k),20))
    for i in range(len(points)):
        otus = np.argmax(sample_names[i])
        assignment = phi.T[i].argsort()[-2:][::-1]
        cluster_1 = assignment[0]
        cluster_2 = assignment[1]
        count[cluster_1*len(mu) + cluster_2][otus] = count[cluster_1*len(mu) + cluster_2][otus] + 1

    print "majority"
    count = count.T
    for i in range(len(count)):
        for j in range(len(count[0]) - 1):
            cluster_1 = j/len(mu)
            cluster_2 = j%len(mu)
            if cluster_1 != cluster_2:
                print '{} & '.format(int(count[i][j])),
        print '{} \\\\ \hline '.format(int(count[i][len(count[0]) - 1]))


    count = np.zeros((int(k*k),20))
    for i in range(len(points)):
        otus = np.argsort(sample_names[i])[-2:][::-1]
        assignment = phi.T[i].argsort()[-2:][::-1]
        cluster_1 = assignment[0]
        cluster_2 = assignment[1]
        count[cluster_1*len(mu) + cluster_2][otus[0]] = count[cluster_1*len(mu) + cluster_2][otus[0]] + 1
        count[cluster_2*len(mu) + cluster_1][otus[0]] = count[cluster_2*len(mu) + cluster_1][otus[0]] + 1
        count[cluster_1*len(mu) + cluster_2][otus[1]] = count[cluster_1*len(mu) + cluster_2][otus[1]] + 1
        count[cluster_2*len(mu) + cluster_1][otus[1]] = count[cluster_2*len(mu) + cluster_1][otus[1]] + 1
    count = count.T
    for i in range(len(count)):
        for j in range(len(count[0]) - 1):
            cluster_1 = j/len(mu)
            cluster_2 = j%len(mu)
            if cluster_1 < cluster_2:
                print '{} & '.format(int(count[i][j])),
        print '{} \\\\ \hline '.format(int(count[i][len(count[0]) - 1]))




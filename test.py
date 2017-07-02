import pandas as pd
import numpy as np
import TwoWaykmeans
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def error_rates(mu_star, u_star, fx_star, C, phi):
    c1 = np.max(phi.T, axis=1)
    y = np.zeros((len(X),2))

    error = 0
    for i in range(len(X)):
        indices = np.nonzero(phi.T[i])
        y[i] = indices[0]

        if y[i][0] != fx_star[i][0] or y[i][1] != fx_star[i][1]:
            error = error + 1

    return float(error)/float(len(X)), np.linalg.norm(C.T - mu_star), np.linalg.norm(u_star - c1)/float(len(X))

if __name__ == '__main__':
    k = 3
    n = 250
    d = 2
    NUM_ROUNDS = 1
    mu = np.array([[0,0],[1,1], [2,0]])
    X = np.zeros((n,d))
    mu_star = mu
    u_star = np.zeros((len(X),1))
    fx_star = np.zeros((len(X),2))

    c1 = []

    for i in range(n):
        j = np.random.choice(k, 2, replace=False, p=[0.2, 0.5, 0.3])
        u = np.random.uniform()
        u_star[i][0] = max(u, 1 - u)
        if j[0] < j[1]:
            fx_star[i][0] = j[0]
            fx_star[i][1] = j[1]
        else:
            fx_star[i][1] = j[0]
            fx_star[i][0] = j[1]

        arr = [0,0,0]
        arr[j[0]] = u
        arr[j[1]] = 1 - u
        c1.append((arr[0], arr[1], arr[2]))

        cov = np.array([[0.001,0],[0,0.001]])
        X[i] = np.random.multivariate_normal(u*mu[j[0]] + (1 - u)*mu[j[1]], cov)

    plt.scatter(X.T[0], X.T[1], c=c1)
    plt.scatter(mu.T[0], mu.T[1], c='w', marker='*', s=200)
    plt.show()


    C, phi = TwoWaykmeans.TwoWayCluster(X,k,NUM_ROUNDS)
    c1 = []
    for i in range(len(phi.T)):
        c1.append((phi.T[i][0], phi.T[i][1], phi.T[i][2]))
    plt.scatter(X.T[0], X.T[1], c=c1)
    plt.scatter(mu.T[0], mu.T[1], c='w', marker='*', s=200)
    plt.scatter(C[0], C[1], c='w', marker='^', s=100)
    plt.show()

    #err1_tot = 0
    #err2_tot = 0
    #err3_tot = 0
    #for i in range(10):

    NUM_ROUNDS = 10
    C,phi = TwoWaykmeans.TwoWayCluster(X,k,NUM_ROUNDS)
    #err1, err2, err3 = error_rates(mu_star, u_star, fx_star, C, phi)
    #err1_tot = err1_tot + err1
    #err2_tot = err2_tot + err2
    #err3_tot = err3_tot + err3

    #print err1_tot/float(10)
    #print err2_tot/float(10)
    #print err3_tot/float(10)

    #TwoWayEM2.classify(X,3,cov_init_magnitude=.001)
    #C, phi = TwoWayClustering.TwoWayCluster(X,3,10)

    c1 = []
    for i in range(len(phi.T)):
        c1.append((phi.T[i][0], phi.T[i][1], phi.T[i][2]))
    plt.scatter(X.T[0], X.T[1], c=c1)
    plt.scatter(mu.T[0], mu.T[1], c='w', marker='*', s=200)
    plt.scatter(C[0], C[1], c='w', marker='^', s=100)
    plt.show()



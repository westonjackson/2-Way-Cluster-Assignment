import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten

####################
#
# 2-Way Expectation Maximization code
#
#####################

#gradient descent for u in [0,1]
def gradient_ascent(muj1, muj2, sigma1, sigma2, x):
    u = .5
    d = 1.0
    step = .005
    for i in range(20):
        d = uniform_gradient(muj1, muj2, sigma1, sigma2, x, u)
        u = u + step*d
        step = step/np.sqrt(i + 1)
        if u < 0 or u > 1:
            break
    if u > 1:
        u = 1
    if u < 0:
        u = 0
    return u

#get gradient for u
def uniform_gradient(muj1, muj2, sigma1, sigma2, x, u):
    deltamu = muj1 - muj2
    deltasig = sigma1 - sigma2
    sigma = u*deltasig + sigma2
    x_minus_mu = x - u*deltamu - muj2

    intermediate = np.linalg.inv(sigma)*deltasig*np.linalg.inv(sigma)
    t1 = -1*np.trace(np.dot(np.linalg.inv(sigma),deltasig))
    t2 = np.dot(np.dot(x_minus_mu,intermediate),x_minus_mu.T)
    t3 = 2*np.dot(np.dot(x_minus_mu,np.linalg.inv(sigma)), deltamu.T)
    return t1 + t2 + t3

#Getting expectation w_ijj (need to vectorize)
def get_expectation(point, u_i, mu, sigma, pi):
    probabilities = np.zeros((1,len(u_i)))
    for i in range(len(u_i)):
        cluster1 = i/len(mu)
        cluster2 = i%len(mu)
        mu_1 = mu[cluster1]
        mu_2 = mu[cluster2]
        sigma_1 = sigma[cluster1]
        sigma_2 = sigma[cluster2]
        u_ijj = u_i[i]
        mean = u_ijj*np.array(mu_1) + (1 - u_ijj)*np.array(mu_2)
        cov = u_ijj*sigma_1 + (1 - u_ijj)*sigma_2
        p = multivariate_normal.pdf(point, mean=mean, cov=cov, allow_singular=True)
        probabilities[0][i] = pi[i]*p
        if u_ijj == 0:
            probabilities[0][i] = 0
        if u_ijj == 1 and cluster1 != cluster2:
            probabilities[0][i] = 0

    return probabilities

#Getting expectation for w_ijj (need to vectorize)
def expectation(points, w, u, mu, sigma, pi):
    g = np.zeros((1,len(w[0])))
    for i in range(len(w)):
        probabilities = get_expectation(points[i], u[i], mu, sigma, pi)
        w[i] = probabilities/(np.sum(probabilities))
        if np.sum(probabilities) == 0:
            w[i] = np.ones((1,len(w[0])))/len(w[0])
        g = g + w[i]
        print g
    return w

#maximize pi
def maximize_pi(w):
    pi = np.sum(w, axis=0)/len(w)
    return pi

#maximize u (need to vectorize)
def maximize_u(points, u, mu, sigma):
    for i in range(len(u)):
        for j in range(len(u[0])):
            cluster_1 = j/len(mu)
            cluster_2 = j%len(mu)
            x = points[i]
            muj1 = np.array(mu[cluster_1])
            muj2 = np.array(mu[cluster_2])
            sigmaj1 = sigma[cluster_1]
            sigmaj2 = sigma[cluster_2]
            if cluster_1 != cluster_2:
                u[i][j] = gradient_ascent(muj1, muj2, sigmaj1, sigmaj2, x)
    return u

#maximize mu
def maximize_mu(points, w, u, mu):
    w_j = np.zeros((len(w), int(np.sqrt(len(w[0])))), dtype=np.float64)
    for i in range(len(mu)):
        w_j.T[i] = w.T[i*len(mu) + i]
    sum_w_j = np.sum(w_j, axis=0)
    mu = np.multiply(np.dot(points.T, w_j),np.reciprocal(sum_w_j))

    return mu.T

#maximize sigma
def maximize_sigma(points, w, u, mu, sigma):
    w_j = np.zeros((len(w), int(np.sqrt(len(w[0])))))
    for i in range(len(mu)):
        w_j.T[i] = w.T[i*len(mu) + i]

    sum_w_j = np.sum(w_j, axis=0)

    for i in range(len(sigma)):
        x_minus_mu = points - mu[i]
        x_minus_mu_w_ijj = (x_minus_mu.T * w_j[:,i]).T
        result = np.dot(x_minus_mu.T,x_minus_mu_w_ijj)
        sigma[i] = result * np.reciprocal(sum_w_j[i])
    return sigma


if __name__ == '__main__':
    num = 1000
    data = pd.read_table("Complete.OTU.by.sample.table.txt", header=0, nrows=num, delimiter="\t")

    labels = data["Group"]
    data.drop(data.columns[[0]], axis=1, inplace=True)
    points = data.as_matrix()[:,0:10].astype(float)
    d = len(points[0])

    #initialize variables
    k = 5

    mu = kmeans(points, k)[0]
    sigma = np.empty((k,d,d))
    for i in range(len(sigma)):
        sigma[i] = np.identity(d)*100000000
    pi = np.ones((1,k*k))/(k*k)
    pi = pi.flatten()
    w = np.zeros((len(points),len(pi)))
    u = np.ones((len(points),len(pi)))

    print "mu"
    print mu

    for i in range(100):
        u = maximize_u(points, u, mu, sigma)
        w = expectation(points, w, u, mu, sigma, pi)
        pi = maximize_pi(w)
        mu = maximize_mu(points, w, u, mu)
        sigma = maximize_sigma(points, w, u, mu, sigma)
        print "pi"
        print pi
        print "mu"
        print mu


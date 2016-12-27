import numpy as np
import math
import scipy.stats as stats
from scipy.stats import multivariate_normal
from scipy.cluster.vq import kmeans
from scipy.misc import logsumexp

chi2 = stats.chi2
TWO_WAY_SIGNFICANCE_LEVEL = .99

####################
#
# 2-Way Expectation Maximization code
#
#####################

#gradient descent for u in [0,1]
def gradient_ascent(muj1, muj2, sigma1, sigma2, x):
    u = .5
    d = 1.0
    eta = 1
    step = eta/4.0

    for i in range(20):
        d = uniform_gradient(muj1, muj2, sigma1, sigma2, x, u)
        if d > 0:
            u = u + eta*step
            step = step/2
        if d < 0:
            u = u - eta*step
            step = step/2
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

def mahalanobis_distance(x, mu, sigma):
    return np.dot(np.dot((x - mu), np.linalg.inv(sigma)),(x - mu).T)

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
        p = multivariate_normal.logpdf(point, mean=mean, cov=cov)
        probabilities[0][i] = p

        if cluster1 != cluster2:
            p_1 = mahalanobis_distance(mean, mu_1, sigma_1)
            p_2 = mahalanobis_distance(mean, mu_2, sigma_2)
            closest = min(p_1,p_2)
            for j in range(len(mu)):
                d = mahalanobis_distance(point, mu[j], sigma[j])
                closest = min(d, closest)

            df = len(point) - 1

            if  stats.chi2.cdf(closest, df) > TWO_WAY_SIGNFICANCE_LEVEL:
                probabilities[0][i] = p
            else:
                probabilities[0][i] = np.NINF

    return probabilities

#Getting expectation for w_ijj (need to vectorize)
def expectation(points, w, u, mu, sigma, pi):
    for i in range(len(w)):
        probabilities = get_expectation(points[i], u[i], mu, sigma, pi)
        total = logsumexp(probabilities, b = pi)
        w[i] = np.exp(np.log(pi) + probabilities - total)
        if np.sum(np.exp(probabilities)) == 0:
            w[i] = np.zeros((1,len(w[0])))
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
    mu = np.multiply(np.dot(points.T, w_j),np.reciprocal(sum_w_j)).T

    return mu

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

def classify(X, k, cov_init_magnitude=1, num_rounds=5, two_way_significance_level = .99):
    
    n = len(X)
    d = len(X[0])
    TWO_WAY_SIGNFICANCE_LEVEL = two_way_significance_level
    
    mu = kmeans(X, k)[0]
    sigma = np.empty((k,d,d))
    for i in range(len(sigma)):
        sigma[i] = np.identity(d)*cov_init_magnitude
    pi = np.ones((1,k*k))/(k*k)
    pi = pi.flatten()
    w = np.zeros((n,k*k))
    u = np.ones((n,k*k))
    
    u = maximize_u(X, u, mu, sigma)

    for i in range(num_rounds):
        w = expectation(X, w, u, mu, sigma, pi)
        pi = maximize_pi(w)
        u = maximize_u(X, u, mu, sigma)
        mu = maximize_mu(X, w, u, mu)
        sigma = maximize_sigma(X, w, u, mu, sigma)
        
    return (w,u,pi,mu,sigma)

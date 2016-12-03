import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import scipy.stats as stats
from scipy.stats import multivariate_normal
from scipy.cluster.vq import kmeans
from scipy.misc import logsumexp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D

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

def get_log_likelihood(points, w, u, mu, sigma, pi):
    log_likelihood_sum = 0
    for i in range(len(points)):
        for j in range(len(pi)):
            cluster_1 = j/len(mu)
            cluster_2 = j%len(mu)
            mean = u[i][j]*mu[cluster_1] + (1 - u[i][j])*mu[cluster_2]
            cov = u[i][j]*sigma[cluster_1] + (1 - u[i][j])*sigma[cluster_2]
            term_1 = np.log(pi[j])
            term_2 = 0.5*np.log(np.linalg.det(np.linalg.inv(cov)))
            term_3 = -0.5*np.dot(np.dot(mean - points[i],cov),(mean - points[i]).T)
            log_likelihood_sum += w[i][j]*(term_1 + term_2 + term_3)
    print log_likelihood_sum

#def get_euclidean_pcoa(points):
#    num_trials = 10
#    num_samples = 5000
#    D = np.zeros((len(points),len(points)))
#    for i in range(num_trials):
#        copy_points = points.copy()
#        for j in range(len(points)):
#            current = copy_points[j]
#            prob = current/float(np.sum(current))
#            sample = np.random.choice(len(current), num_samples, p=prob)
#            copy_points[j] = np.bincount(sample, minlength=len(current))
#        D += euclidean_distances(copy_points)
#    D = D/num_trials
#    D = D/D.max()
#    mds = MDS(n_components=3, dissimilarity='precomputed')
#    return mds.fit_transform(D)

if __name__ == '__main__':
    sample_names = pd.read_table("combined.clean.an.0.03.subsample.shared", header=0, delimiter="\t")
    sample_names.drop(sample_names.columns[[0,1,2]], axis=1, inplace=True)
    sample_names = sample_names.as_matrix().astype(float)[:,0:20]


    data = pd.read_table("combined.clean.an.thetayc.0.03.lt.ave.pcoa.axes", header=0, delimiter="\t")
    labels = data["group"]
    data.drop(data.columns[[0]], axis=1, inplace=True)
    points = data.as_matrix().astype(float)
    points = points[:,0:3]


    #data = pd.read_table("Complete.OTU.by.sample.table.txt", header=0, delimiter="\t")
    #labels = data["Group"]
    #data.drop(data.columns[[0]], axis=1, inplace=True)
    #points = data.as_matrix().astype(float)
    #points = points[np.where(np.sum(points,axis=1) > 5000)]
    #points = get_euclidean_pcoa(points)

    n = len(points)
    d = len(points[0])

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(points.T[0], points.T[1], points.T[2])
    #plt.show()

    #training_data = points[:int(.8*n)]
    #test_data = points[int(.8*n):]
    #n = int(.8*n)

    k = 5

    mu = kmeans(points, k)[0]
    sigma = np.empty((k,d,d))
    for i in range(len(sigma)):
        sigma[i] = np.identity(d)*.001
    pi = np.ones((1,k*k))/(k*k)
    pi = pi.flatten()
    w = np.zeros((n,k*k))
    u = np.ones((n,k*k))
    u = maximize_u(points, u, mu, sigma)

    print "mu"
    print mu

    for i in range(1):
        #test_w = np.zeros((len(test_data),k*k))
        #test_u = np.ones((len(test_data),k*k))
        #test_u = maximize_u(test_data, test_u, mu, sigma)
        #d_sum = 0
        #for i in range(len(test_data)):
        #    min_distance = np.inf
        #    for j in range(len(test_u[0])):
        #        cluster1 = j/len(mu)
        #        cluster2 = j%len(mu)
        #        mu_1 = mu[cluster1]
        #        mu_2 = mu[cluster2]
        #        sigma_1 = sigma[cluster1]
        #        sigma_2 = sigma[cluster2]
        #        mean = test_u[i,j]*np.array(mu_1) + (1 - test_u[i,j])*np.array(mu_2)
        #        cov = test_u[i,j]*sigma_1 + (1 - test_u[i,j])*sigma_2
        #        d = np.sqrt(mahalanobis_distance(test_data[i], mean, cov))
        #        if min_distance > d:
        #            min_distance = d
        #    print min_distance
        #    d_sum += min_distance
        #print d_sum/float(len(test_data))
        c1 = np.max(w, axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points.T[0], points.T[1], points.T[2], c=c1)
        ax.scatter(mu.T[0], mu.T[1], mu.T[2], c='r')
        plt.show()

        w = expectation(points, w, u, mu, sigma, pi)
        pi = maximize_pi(w)
        u = maximize_u(points, u, mu, sigma)
        mu = maximize_mu(points, w, u, mu)
        sigma = maximize_sigma(points, w, u, mu, sigma)
        print "pi"
        print pi
        print "mu"
        print mu
        print "sigma"
        print sigma

    c1 = np.max(w, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points.T[0], points.T[1], points.T[2], c=c1)
    ax.scatter(mu.T[0], mu.T[1], mu.T[2], c='r')
    plt.show()

    c2 = np.argmax(w, axis=1)
    count = np.zeros(((int(k*k)),20))
    for i in range(int(k*k)):
        for j in range(len(points)):
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
        for j in range(len(points)):
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




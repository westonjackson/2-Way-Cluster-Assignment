import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


#FOR CREATING DATA POINTS
def create_points(n, mu, sigma, pi):
    points = np.empty([2, 0])
    u = np.empty((0,len(mu)))
    prior_mu = np.empty([2,0])
    for i in range(n):
        point = get_point(mu, sigma, pi)
        points = np.append(points, point[0], axis=1)
        u = np.append(u, point[1], axis=0)
        prior_mu = np.append(prior_mu, point[2], axis=1)
    return (points, u, prior_mu)

#FOR CREATING DATA POINTS
def get_point(mu, sigma, pi):
    i = np.random.choice(np.arange(0, len(pi)), p=pi)
    cluster_1 = i/len(mu)
    cluster_2 = i%len(mu)
    u_ijj = np.zeros((1,len(mu)))
    if cluster_1 == cluster_2:
        mean = mu[cluster_1]
        cov = sigma[cluster_1]
        u_ijj[0][cluster_1] = 1
        return (np.random.multivariate_normal(mean, cov, 1).T, u_ijj, [[cluster_1],[cluster_2]])

    mu_1 = mu[cluster_1]
    mu_2 = mu[cluster_2]
    sigma_1 = sigma[cluster_1]
    sigma_2 = sigma[cluster_2]
    u = np.random.uniform()
    u_ijj[0][cluster_1] = u
    u_ijj[0][cluster_2] = 1 - u

    mean = u*np.array(mu_1) + (1 - u)*np.array(mu_2)
    cov = u*sigma_1 + (1 - u)*sigma_2

    return (np.random.multivariate_normal(mean, cov, 1).T, u_ijj, [[cluster_1],[cluster_2]])

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
    for i in range(len(w)):
        probabilities = get_expectation(points[i], u[i], mu, sigma, pi)
        w[i] = probabilities/(np.sum(probabilities))
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

def graph(points, mu, sigma):
    plot_mu = np.array(mu)
    ax = plt.subplot(111,aspect='auto')
    for i in range(k):
        mean = plot_mu[i]
        cov = sigma[i]
        lambda_, v = np.linalg.eig(cov)
        ell = Ellipse(xy=(mean[0],mean[1]),width=np.sqrt(lambda_[0]), height=np.sqrt(lambda_[1]), angle=np.rad2deg(np.arccos(v[0,0])))
        ell.set_facecolor('none')
        ax.add_artist(ell)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)


    c1 = np.max(w, axis=1)

    plt.scatter(points.T[0], points.T[1], c=c1)
    plt.scatter(mu.T[0], mu.T[1], color='red')
    plt.show()

def error_calculations(points, w, u, mu, sigma, pi):
    posterior_mu = np.argmax(w, axis=1)
    correct = 0
    partially_correct = 0
    for i in range(len(points)):
        if prior_mu[i][0] == posterior_mu[i] / len(mu) and prior_mu[i][1] == posterior_mu[i] % len(mu):
            correct = correct + 1
            partially_correct = partially_correct + 1
        elif prior_mu[i][0] == posterior_mu[i] % len(mu) and prior_mu[i][1] == posterior_mu[i] / len(mu):
            correct = correct + 1
            partially_correct = partially_correct + 1
        else:
            if prior_mu[i][0] == posterior_mu[i] / len(mu) or prior_mu[i][1] == posterior_mu[i] % len(mu):
                partially_correct = partially_correct + 1
            elif prior_mu[i][0] == posterior_mu[i] % len(mu) or prior_mu[i][1] == posterior_mu[i] / len(mu):
                partially_correct = partially_correct + 1


    print "cluster error rate"
    print 1 - correct/float(len(points))

    print "partial cluster error rate"
    print 1 - partially_correct/float(len(points))

    #posterior_u = np.multiply(u,w)
    total = 0
    for i in range(len(u)):

        posterior_u = np.zeros(np.shape(prior_u[i]))

        if posterior_mu[i]/len(mu) == posterior_mu[i]%len(mu):
            posterior_u[posterior_mu[i]/len(mu)] = 1
        else:
            cluster_1 = posterior_mu[i]/len(mu)
            cluster_2 = posterior_mu[i]%len(mu)
            posterior_u[cluster_1] = u[i][posterior_mu[i]]
            posterior_u[cluster_2] = u[i][cluster_2*len(mu) + cluster_1]
        print "current u and prior"
        print posterior_u
        print prior_u[i]

        total += np.sum(np.square(prior_u[i] - posterior_u))
    print "average squared distance per point"
    print total/(len(points))


if __name__ == '__main__':
    mu = [np.array([5,0]), np.array([5,5]),np.array([10,5])]
    sigma = [np.array([[.2,0],[0,.2]]),np.array([[.4,0],[0,.4]]),np.array([[.1,0],[0,.1]])]
    pi = [.3,.025,.025,.025,.25,.025,.025,.025,.3]

    NUM_POINTS = 500
    result = create_points(NUM_POINTS, mu, sigma, pi)
    points = result[0].T
    prior_u = result[1]
    prior_mu = result[2].T


    d = len(points[0])
    num = len(points)

    k = 3
    idx = np.random.randint(num, size=k)
    mu = points[idx,:]
    sigma = np.empty((k,d,d))
    for i in range(len(sigma)):
        sigma[i] = np.identity(d)
    pi = (np.ones((1,k*k))/(k*k))
    pi = pi.flatten()
    print mu
    print sigma
    print pi

    w = np.zeros((len(points),len(pi)))
    u = np.ones((len(points),len(pi)))
    for i in range(100):
        u = maximize_u(points, u, mu, sigma)
        w = expectation(points, w, u, mu, sigma, pi)
        pi = maximize_pi(w)
        mu = maximize_mu(points, w, u, mu)
        sigma = maximize_sigma(points, w, u, mu, sigma)
        print i
        print pi
        print mu
        print sigma
        graph(points, mu, sigma)
        error_calculations(points, w, u, mu, sigma, pi)
    error_calculations(points, w, u, mu, sigma, pi)
    graph(points, mu, sigma)




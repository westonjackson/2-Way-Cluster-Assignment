import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


#FOR CREATING DATA POINTS
def create_points(n, mu, sigma, pi, dic):
    points = np.empty([2, 0])
    for i in range(n):
        point = get_point(mu, sigma, pi, dic)
        points = np.append(points, point, axis=1)
    return points

#FOR CREATING DATA POINTS
def get_point(mu, sigma, pi, dic):
    index = np.random.choice(np.arange(0, len(pi)), p=pi)
    clusters = dic[index]
    if clusters[0] == clusters[1]:
        mean = mu[clusters[0]]
        cov = sigma[clusters[0]]
        return np.random.multivariate_normal(mean, cov, 1).T

    mu_1 = mu[clusters[0]]
    mu_2 = mu[clusters[1]]
    sigma_1 = sigma[clusters[0]]
    sigma_2 = sigma[clusters[1]]
    u = np.random.uniform()

    mean = u*np.array(mu_1) + (1 - u)*np.array(mu_2)
    cov = u*sigma_1 + (1 - u)*sigma_2

    return np.random.multivariate_normal(mean, cov, 1).T

#Used to map indices to tuples, 0 -> (1,1)
def get_map(n):
    dic = {}
    j = 0
    k = 0
    for i in range(n):
        j = i
        while i <= j and j < n:
            dic[k] = [i,j]
            j+=1
            k+=1
    return dic

#helper function to go from (1,1) -> 0
def get_index(xy, n):
    x = xy[0]
    y = xy[1]
    return int(0.5*(x - 1)*(2*n - x + 2) + y - x)

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
def get_expectation(point, u_i, mu, sigma, pi, dic):
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
        probabilities[0][i] = p
        if u_ijj == 0:
            probabilities[0][i] = 0
        if u_ijj == 1 and cluster1 != cluster2:
            probabilities[0][i] = 0

    return probabilities

#Getting expectation for w_ijj (need to vectorize)
def expectation(points, w, u, mu, sigma, pi, dic):
    for i in range(len(w)):
        probabilities = get_expectation(points[i], u[i], mu, sigma, pi, dic)
        w[i] = probabilities/(np.sum(probabilities))
    return w

#maximize pi
def maximize_pi(w):
    pi = np.sum(w, axis=0)/len(w)
    return pi

#maximize u (need to vectorize)
def maximize_u(points, u, mu, sigma, dic):
    for i in range(len(u)):
        for j in range(len(u[0])):
            cluster_1 = j/3
            cluster_2 = j%3
            x = points[i]
            muj1 = np.array(mu[cluster_1])
            muj2 = np.array(mu[cluster_2])
            sigmaj1 = sigma[cluster_1]
            sigmaj2 = sigma[cluster_2]
            if cluster_1 != cluster_2:
                u[i][j] = gradient_ascent(muj1, muj2, sigmaj1, sigmaj2, x)
    return u

#maximize mu (need to vectorize)
def maximize_mu(points, w, u, mu, dic):
    w_j = np.zeros((len(w), int(np.sqrt(len(w[0])))), dtype=np.float64)
    for i in range(len(mu)):
        w_j.T[i] = w.T[i*len(mu) + i]
    mu = np.zeros(np.shape(mu))
    sum_w_j = np.sum(w_j, axis=0)
    for i in range(len(mu)):
        for j in range(len(points)):
            mu[i] = mu[i] + points[j]*w_j[j][i]
        mu[i] = mu[i] * np.reciprocal(sum_w_j[i])

    return mu

#maximize sigma (need to vectorize)
def maximize_sigma(points, w, u, mu, sigma, dic):
    w_j = np.zeros((len(w), int(np.sqrt(len(w[0])))))

    for i in range(len(mu)):
        w_j.T[i] = w.T[i*len(mu) + i]

    sum_w_j = np.sum(w_j, axis=0)
    for i in range(len(sigma)):
        x_minus_mu = points - mu[i]
        sigma[i] = np.zeros(np.shape(sigma[i]))
        for j in range(len(x_minus_mu)):
            temp = np.array([x_minus_mu[j]]).T * np.array([x_minus_mu[j]])
            temp = temp * w_j[j][i]
            sigma[i] = sigma[i] + temp
        sigma[i] = sigma[i] * np.reciprocal(sum_w_j[i])

    return sigma

if __name__ == '__main__':
    mu = [np.array([0,1]), np.array([5,5]),np.array([10,0])]
    sigma = [np.array([[.5,0],[0,.5]]),np.array([[.5,0],[0,.5]]),np.array([[.5,0],[0,.5]])]
    pi = [.3,.05,.05,.25,.05,.3]
    dic = get_map(len(mu))

    NUM_POINTS = 1000
    points = create_points(NUM_POINTS, mu, sigma, pi, dic).T

    mu = [[0,0], [0,1], [1,0]]
    sigma = [np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])]
    pi = [1,0,0,0,1,0,0,0,1]
    k = 3

    w = np.zeros((len(points),len(pi)))
    u = np.ones((len(points),len(pi)))
    for i in range(5):
        u = maximize_u(points, u, mu, sigma, dic)
        w = expectation(points, w, u, mu, sigma, pi, dic)
        pi = maximize_pi(w)
        mu = maximize_mu(points, w, u, mu, dic)
        sigma = maximize_sigma(points, w, u, mu, sigma, dic)
        print mu

    mu = np.array(mu)
    ax = plt.subplot(111,aspect='auto')
    for i in range(k):
        mean = mu[i]
        cov = sigma[i]
        lambda_, v = np.linalg.eig(cov)
        ell = Ellipse(xy=(mean[0],mean[1]),width=np.sqrt(lambda_[0]), height=np.sqrt(lambda_[1]), angle=np.rad2deg(np.arccos(v[0,0])))
        ell.set_facecolor('none')
        ax.add_artist(ell)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    plt.scatter(points.T[0], points.T[1])
    plt.scatter(mu.T[0], mu.T[1], color='red')
    plt.show()

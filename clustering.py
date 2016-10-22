import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import multivariate_normal

def create_points(n, mu, sigma, pi, dic):
    points = np.empty([2, 0])
    for i in range(n):
        point = get_point(mu, sigma, pi, dic)
        points = np.append(points, point, axis=1)
    return points

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

def gradient_ascent(muj1, muj2, sigma1, sigma2, x):
    u = .5
    d = 1.0
    step = .1
    while d > .0001:
        d = uniform_gradient(muj1, muj2, sigma1, sigma2, x, u)
        u = u + step*d
        if u < 0 or u > 1:
            break
    if u > 1:
        u = 1
    if u < 1:
        u = 0
    return u

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


def get_expectation(point, u_i, mu, sigma, pi, dic):
    probabilities = np.zeros((1,len(u_i)),dtype=np.float64)
    for i in range(len(u_i)):
        clusters = dic[i]
        mu_1 = mu[clusters[0]]
        mu_2 = mu[clusters[1]]
        sigma_1 = sigma[clusters[0]]
        sigma_2 = sigma[clusters[1]]
        u_ijj = u_i[i]
        mean = u_ijj*np.array(mu_1) + (1 - u_ijj)*np.array(mu_2)
        cov = u_ijj*sigma_1 + (1 - u_ijj)*sigma_2
        p = multivariate_normal.pdf(point, mean=mean, cov=cov)
        probabilities[0][i] = p
    return probabilities


def expectation(points, w, u, mu, sigma, pi, dic):
    for i in range(len(w)):
        probabilities = get_expectation(points[i], u[i], mu, sigma, pi, dic)
        w[i] = probabilities/np.sum(probabilities)
    return w

def maximize_pi(w):
    pi = np.sum(w, axis=0)/len(w)
    return pi

def maximize_u(points, u, mu, sigma, dic):
    for i in range(len(u)):
        for j in range(len(u[0])):
            clusters = dic[j]
            x = points[i]
            muj1 = mu[clusters[0]]
            muj2 = mu[clusters[1]]
            sigmaj1 = sigma[clusters[0]]
            sigmaj2 = sigma[clusters[1]]
            if clusters[0] != clusters[1]:
                u[i][j] = gradient_ascent(muj1, muj2, sigmaj1, sigmaj2, x)
    return u

def maximize_mu(points, w, u, mu, dic):
    temp = mu
    for i in range(len(mu)):
        top = np.zeros((1,2))
        bottom = 0
        for j in range(len(points)):
            for k in range(len(mu)):
                t1 = min(i,k)
                t2 = max(i,k)
                index = get_index((t1 + 1,t2 + 1), len(mu))
                x = points[j]
                w_ijj = w[j][index]
                u_ijj = u[j][index]
                if k < i:
                    u_ijj = 1 - u_ijj

                muj2 = mu[k]
                muj1 = mu[i]
                top += w_ijj*u_ijj*(x - muj2*(1 - u_ijj))
                bottom += w_ijj*u_ijj*u_ijj
        temp[i] = np.array(top/bottom)[0]
    mu = temp
    return mu

def maximize_sigma(points, w, u, mu, sigma, dic):
    temp = sigma
    for i in range(len(sigma)):
        top = np.zeros((2,2))
        bottom = 0
        for j in range(len(points)):
            for k in range(len(sigma)):
                t1 = min(i,k)
                t2 = max(i,k)

                index = get_index((t1 + 1,t2 + 1), len(mu))
                x = points[j]
                w_ijj = w[j][index]
                u_ijj = u[j][index]
                if k < i:
                    u_ijj = 1 - u_ijj
                muj2 = mu[k]
                muj1 = mu[i]
                sigmaj1 = sigma[k]
                sigmaj2 = sigma[i]
                mean = [u_ijj*muj1 + (1 - u_ijj)*muj2]
                top += w_ijj*((x - mean).T*(x - mean) + u_ijj*(1 - u_ijj)*sigmaj2)
                bottom += w_ijj*u_ijj*u_ijj
        temp[i] = top/bottom
    sigma = temp
    return sigma

if __name__ == '__main__':
    mu = [np.array([50,0]),np.array([50,50]),np.array([100,0])]
    sigma = [np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])]
    pi = [.2,.2,.1,.2,.1,.2]
    dic = get_map(len(mu))

    NUM_POINTS = 100
    points = create_points(NUM_POINTS, mu, sigma, pi, dic).T

    mu = [np.array([50,0]),np.array([50,50]),np.array([100,0])]
    sigma = [np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])]
    pi = [1,0,0,1,0,1]
    k = 3

    w = np.zeros((len(points),6))
    u = np.ones((len(points),6))
    for i in range(5):
        w = expectation(points, w, u, mu, sigma, pi, dic)
        pi = maximize_pi(w)
        print pi
        print u
        u = maximize_u(points, u, mu, sigma, dic)
        mu = maximize_mu(points, w, u, mu, dic)
        print mu
        sigma = maximize_sigma(points, w, u, mu, sigma, dic)
        print sigma

    mu = np.array(mu)
    plt.scatter(mu.T[0], mu.T[1], color='red')
    plt.scatter(points.T[0], points.T[1])
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

def create_points(n, mu, sigma, pi, dic):
    points = np.empty([2, 0])
    for i in range(n):
        point = get_point(mu, sigma, pi, dic)
        points = np.append(points, point, axis=1)
    return points

def get_point(mu, sigma, pi, dic):
    index = np.random.choice(np.arange(1, len(pi) + 1), p=pi)
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
    k = 1
    for i in range(n):
        j = i
        while i <= j and j < n:
            dic[k] = [i,j]
            j+=1
            k+=1
    return dic



if __name__ == '__main__':
    mu = [np.array([1,1]),np.array([10,10]),np.array([10,1])]
    sigma = [np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])]
    pi = [.2,.05,.05,.2,.1,.4]
    dic = get_map(len(mu))

    NUM_POINTS = 1000
    points = create_points(NUM_POINTS, mu, sigma, pi, dic)

    plt.scatter(points[0], points[1])
    plt.show()


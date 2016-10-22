import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

def same_covariance_estimate(muj1, muj2, sigma1, sigma2, x):
    diff = muj1 - muj2
    x_diff = x - muj2
    top = np.dot(np.dot(diff, np.linalg.inv(sigma1)), x_diff.T)
    bottom = np.dot(np.dot(diff, np.linalg.inv(sigma1)), diff.T)
    return top/float(bottom)


def gradient_ascent(muj1, muj2, sigma1, sigma2, x):
    u = .5
    d = 1.0
    step = .1
    while d > .0001:
        d = uniform_gradient(muj1, muj2, sigma1, sigma2, x, u)
        u = u + step*d
        if u < 0:
            u = 0
            break
        if u > 1:
            u = 1
            break
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

if __name__ == '__main__':
    muj1 = np.array([[0,0]])
    muj2 = np.array([[1,1]])
    sigma1 = np.array([[1,0],[0,1.2]])
    sigma2 = np.array([[1.3,0],[0,1.0]])
    x = np.array([[0.5,0.5]])

    n = 10000

    normal_results = np.zeros((2,n), dtype=np.float32)
    for i in range(n):
        u = np.random.uniform()
        mean = u*muj1 + (1 - u)*muj2
        cov = u*sigma1 + (1 - u)*sigma2

        #term one
        det = np.linalg.det(cov)
        term_one = -0.5*np.log(det)

        #term two
        diff = x - mean
        term_two = -0.5 * np.dot(np.dot(diff, np.linalg.inv(cov)),diff.T)
        result = term_one + term_two

        normal_results[0,i] = u
        normal_results[1,i] = result[0][0]

    index = np.argmax(normal_results[1])
    print "Max value for normal"
    print normal_results[0,index]
    print "Same covariance"
    print same_covariance_estimate(muj1, muj2, sigma1, sigma2, x)[0][0]
    print "gradient function"
    print gradient_ascent(muj1, muj2, sigma1, sigma2, x)

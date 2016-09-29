import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

if __name__ == '__main__':
    muj1 = np.array([[0,0]])
    muj2 = np.array([[1,0]])
    sigma1 = np.array([[1,0],[0,1]])
    sigma2 = np.array([[1,0],[0,1]])
    x = np.array([[0.5,0.5]])

    n = 100

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

    plt.scatter(normal_results[0], normal_results[1])
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import scipy.stats as stats
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from scipy.cluster.vq import kmeans
from scipy.misc import logsumexp
chi2 = stats.chi2

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

def objective_value(muj1, muj2, sigma1, sigma2, x, u):
    mean = u*muj1 + (1 - u)*muj2
    sigma = u*sigma1 + (1 - u)*sigma2
    return multivariate_normal.logpdf(x,mean,sigma)

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
    for i in range(len(probabilities[0])):
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
            
            if  stats.chi2.cdf(closest, df) < .99:
                probabilities[0][i] = np.NINF
            else:
                probabilities[0][i] = p

    return probabilities


#Getting expectation for w_ijj (need to vectorize)
def expectation(points, w, u, mu, sigma, pi):
    for i in range(len(points)):
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
            

def error_calculations(points, w, u, mu, sigma, pi, prior_mu, prior_u, mu_0, sigma_0, pi_0):
    posterior_mu = np.argmax(w, axis=1)
    correct = 0
    for i in range(len(points)):
        if prior_mu[i][0] == posterior_mu[i] / len(mu) and prior_mu[i][1] == posterior_mu[i] % len(mu):
            correct = correct + 1
        elif prior_mu[i][0] == posterior_mu[i] % len(mu) and prior_mu[i][1] == posterior_mu[i] / len(mu):
            correct = correct + 1


    print "cluster error rate"
    cluster_error = 1 - correct/float(len(points))
    print cluster_error

    print "squared difference from pi"
    pi_diff = np.sum(np.square(pi - pi_0))
    print pi_diff

    print "squared difference from mu"
    mu_diff = np.sum(np.square(mu - mu_0))
    print mu_diff

    print "squared difference from sigma"
    sigma_diff = np.sum(np.square(sigma - sigma_0))
    print sigma_diff

    new_total = 0
    for i in range(len(u)):
        posterior_u = np.zeros(np.shape(prior_u[i]))

        if posterior_mu[i]/len(mu) == posterior_mu[i]%len(mu):
            posterior_u[posterior_mu[i]/len(mu)] = 1
        else:
            cluster_1 = posterior_mu[i]/len(mu)
            cluster_2 = posterior_mu[i]%len(mu)
            posterior_u[cluster_1] = u[i][posterior_mu[i]]
            posterior_u[cluster_2] = u[i][cluster_2*len(mu) + cluster_1]
        new_total += np.sum(np.square(prior_u[i] - posterior_u))
    print "average squared assignment per point (with u)"
    avg_error = new_total/(len(points))
    print avg_error

    return (cluster_error, pi_diff, mu_diff, sigma_diff, avg_error)



if __name__ == '__main__':
    mu_0 = [np.array([0,0]), np.array([5,5]),np.array([10,0])]
    sigma_0 = [np.array([[.1,0],[0,.1]]),np.array([[.2,0],[0,.2]]),np.array([[.1,0],[0,.1]])]
    pi_0 = [.25,.05,.025,.05,.25,.05,.025,.05,.25]

    k = 3
    d = len(mu_0[0])

    NUM_POINTS = 250
    c_error = 0
    p_error = 0
    m_error = 0
    s_error = 0
    a_error = 0
    pi_f  = np.zeros((1,k*k))/(k*k)
    mu_f = np.zeros((k,d))
    sigma_f  = np.zeros((k,d,d))
    for i in range(10):
        result = create_points(NUM_POINTS, mu_0, sigma_0, pi_0)
        points = result[0].T
        prior_u = result[1]
        prior_mu = result[2].T
        
        num = len(points)

        idx = np.random.randint(num, size=k)
        means = kmeans(points, k)[0]
        mu = means
        mu = mu[mu[:,0].argsort()] #for error checking
        sigma = np.empty((k,d,d))
        for j in range(len(sigma)):
            sigma[j] = np.identity(d)
        pi = np.ones((1,k*k))/(k*k)
        pi = pi.flatten()
        w = np.zeros((len(points),len(pi)))
        u = np.ones((len(points),len(pi)))
        u = maximize_u(points, u, mu, sigma)

        
        for j in range(10):
            w = expectation(points, w, u, mu, sigma, pi)
            print get_log_likelihood(points, w, u, mu, sigma, pi)
            pi = maximize_pi(w)
            u = maximize_u(points, u, mu, sigma)
            mu = maximize_mu(points, w, u, mu)
            sigma = maximize_sigma(points, w, u, mu, sigma)
            print pi
            print mu
            print sigma
            error = error_calculations(points, w, u, mu, sigma, pi, prior_mu, prior_u, mu_0, sigma_0, pi_0)
            graph(points, mu, sigma)
            
        error = error_calculations(points, w, u, mu, sigma, pi, prior_mu, prior_u, mu_0, sigma_0, pi_0)
        c_error += error[0]
        p_error += error[1]
        m_error += error[2]
        s_error += error[3]
        a_error += error[4]
        pi_f += pi
        mu_f += mu
        sigma_f += sigma
        print "ERROR"
        print c_error
        print p_error
        print m_error
        print s_error
        print a_error
        print pi_f
        print mu_f
        print sigma_f

    print pi_f/10
    print mu_f/10
    print sigma_f/10
    print c_error/10
    print p_error/10
    print m_error/10
    print s_error/10
    print a_error/10




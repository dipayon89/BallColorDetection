import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(1000)  # For debugging and reproducibility

N = 1000

pi1 = 0.15
pi2 = 0.35
pi3 = 0.5

mu1 = np.array([-3.0, 2.0])
mu2 = np.array([4.0, 2.0])
mu3 = np.array([3.0, 1.0])

Sigma1 = np.array([
    [1.0, 0.2],
    [0.2, 2.0],
])
Sigma2 = np.array([
    [0.8, -1.0],
    [-1.0, 1.3],
])
Sigma3 = np.array([
    [1.5, 1.4],
    [1.4, 1.5],
])

Z_seeds = np.random.uniform(0, 1, size=N)

print(Z_seeds)

X = np.empty((0, 2))
for z_seed in Z_seeds:
    if z_seed < pi1:
        x_n = np.random.multivariate_normal(
            mean=mu1,
            cov=Sigma1,
            size=1)
    elif z_seed < pi1 + pi2:
        x_n = np.random.multivariate_normal(
            mean=mu2,
            cov=Sigma2,
            size=1)
    else:
        x_n = np.random.multivariate_normal(
            mean=mu3,
            cov=Sigma3,
            size=1)
    X = np.vstack((X, x_n))

plt.plot(X[:, 0], X[:, 1], "o")
plt.show()

# Initialize parameters
class theta:
    pi = np.empty((0, 3))
    mu = np.empty((0, 3, 2))
    Sigma = np.empty((0, 3, 2, 2))

    def __init__(self, pi, mu, Sigma):
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma


theta_old = theta(
    pi=np.array([0.4, 0.3, 0.3]),
    mu=np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ]),
    Sigma=np.array([
        [[1.0, 0.5], [0.5, 1.0]],
        [[1.0, 0.5], [0.5, 1.0]],
        [[1.0, 0.5], [0.5, 1.0]]
    ])
)


def get_gamma(theta_val):
    gamma = np.empty((0, 3))

    for n in range(N):
        gamma_n = np.array([])

        for k in range(3):
            denom_elem = np.array([])
            for j in range(3):
                dist_j = multivariate_normal(
                    mean=theta_val.mu[j],
                    cov=theta_val.Sigma[j])
                denom_elem = np.append(denom_elem, theta_val.pi[j] * dist_j.pdf(X[n]))
            denom = denom_elem.sum()

            dist_k = multivariate_normal(
                mean=theta_val.mu[k],
                cov=theta_val.Sigma[k])
            numer = theta_val.pi[k] * dist_k.pdf(X[n])

            gamma_n = np.append(gamma_n, numer / denom)

        gamma = np.vstack((gamma, gamma_n))

    return gamma


def get_mu(gamma):
    mu_new = np.zeros((3, 2), dtype=np.float64)

    for k in range(3):
        denom = sum(gamma[n][k] for n in range(N))
        numer_x = sum(gamma[n][k] * X[n][0] for n in range(N))
        mu_new[k][0] = numer_x / denom
        numer_y = sum(gamma[n][k] * X[n][1] for n in range(N))
        mu_new[k][1] = numer_y / denom

    return mu_new


def get_sigma(gamma, mu_new):
    sigma_new = np.empty((0, 2, 2))

    for k in range(3):
        denom = sum(gamma[n][k] for n in range(N))
        numer = np.zeros((2, 2), dtype=np.float64)
        for n in range(N):
            sub = np.subtract(X[n], mu_new[k])
            sub = np.array([sub])
            sub_t = sub.transpose()
            numer = numer + gamma[n][k] * np.matmul(sub_t, sub)
        sigma_new = np.vstack((sigma_new, [numer / denom]))

    return sigma_new


def get_pi(gamma):
    pi_new = np.array([])

    for k in range(3):
        pi_new = np.append(
            pi_new,
            sum(gamma[n][k] for n in range(N)) / N)

    return pi_new


for loop in range(50):
    print("Running iteration {} ...".format(loop + 1), end="\r")
    # Get gamma
    l_gamma = get_gamma(theta_old)
    # Get new mu
    l_mu_new = get_mu(l_gamma)
    # Get new sigma
    l_sigma_new = get_sigma(l_gamma, l_mu_new)
    # Get new pi
    l_pi_new = get_pi(l_gamma)
    # Replace theta
    theta_old = theta(
        pi=l_pi_new,
        mu=l_mu_new,
        Sigma=l_sigma_new
    )

print("\nDone")

print(theta_old.pi)

print(theta_old.mu)

print(theta_old.Sigma)

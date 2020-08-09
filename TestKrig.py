import numpy as np

from EI_problem import eim_eu, eim_maxmin

if __name__ == "__main__":
    a = np.array([[1,7], [4,0]])
    b = np.sqrt(a)
    print(b)

    mu = np.loadtxt('mu.csv', delimiter=',')
    s = np.loadtxt('sig.csv', delimiter=',')
    f = np.loadtxt('nd.csv', delimiter=',')

    mu = np.atleast_2d(mu)
    s = np.atleast_2d(s)
    f = np.atleast_2d(f)
    out = eim_maxmin(mu, s, f, np.atleast_2d([1.1, 1.1]))
    print(out)



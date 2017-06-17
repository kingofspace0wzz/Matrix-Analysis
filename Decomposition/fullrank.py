__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_rank as rank


# return full-rank decomposition of X = FG^T
def fullrank(X):

    rankX = rank(X)

    U, eigvals, Vh = la.svd(X)

    #construct a r-rank sigma-square-root matrix

    sigma = np.eye(rankX)
    for i in range(sigma.shape[0]):
        sigma[i, i] = np.sqrt(eigvals[i])

    F = U.dot(np.vstack((sigma, np.zeros((X.shape[0] - rankX, rankX)))))
    Gh = np.hstack((sigma, np.zeros((rankX, X.shape[1] - rankX)))).dot(Vh)

    return F, Gh



def test():

    A = np.array([[1,3,2,1,4],
                  [2,6,1,0,7],
                  [3,9,3,1,11]])

    print('A: ', A, '\n', 'F: ', fullrank(A)[0], '\n', 'G: ', fullrank(A)[1])
    print('\n', 'FG: ', fullrank(A)[0].dot(fullrank(A)[1]))

if __name__ == '__main__':
    test()

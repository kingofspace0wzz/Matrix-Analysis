__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from Matrix.SpeicalMatrix import isSymmetric
'''
    Methods of eigenvalue approximation


'''


def power(A, x, tol = 0.0001, N =5000, accelerate = True):
    """
    return the dominant eigenvalue(the one associated with the largest magnitude) of matrix A,
    disadvantage: does not know whether matrix A has a dominant eigenvalue

        tol: tolerance

        N: maximum number of iterations

    """

    p = -1

    for i in range(A.shape[1]):
        p += 1
        if la.norm(x, np.inf) == x[i]:
            break

    x = x / x[p]

    for k in range(N):

        y = np.dot(A, x)

        u = y[p]

        p = -1

        for i in range(len(y)):
            p += 1
            if la.norm(y, np.inf) == y[i]:
                break

        if y[p] == 0:
            raise Exception("A has the eigenvalue 0, select a new vector x and restart")

        err = la.norm(x-(y/y[p]), np.inf)

        x = y/y[p]

        if err < tol:
            return u, x

    raise Exception("number of iterations exceeded")

def power_symmetric(A, x, tol = 0.0001, N =5000):
    """
    return the dominant eigenvalue(the one associated with the largest magnitude) of a SYMMETRIC matrix A,
    disadvantage: does not know whether matrix A has a dominant eigenvalue

        tol: tolerance
        N: maximum number of iterations

    """
    # check if matrix A is Symmetric
    if isSymmetric(A) != True:
        raise Exception("Matrix is not symmetric.")

    x = x/la.norm(x, 2)

    for i in range(N):
        y = np.dot(A, x)
        u = np.dot(x, y)
        if abs(la.norm(y, 2) - 0) < 0.0001:
            raise Exception("A has the eigenvalue 0, select a new vector x and restart")

        err = la.norm(x - y/la.norm(y), 2)
        x = y/la.norm(y, 2)
        if err < tol:
            return u, x


    raise Exception("number of iterations exceeded")


def power_inverse(A, x, tol = 0.0001, N =5000):
    '''
    inverse power method
    faster convergence rate

        tol: tolerance
        N: maximum number of iterations

    '''

    q = np.dot(x.T, np.dot(A, x))/np.dot(x.T, x)

    p = -1
    u0, u1 = 0, 0
    for i in range(A.shape[1]):
        p += 1
        if la.norm(x, np.inf) == x[i]:
            break

    x = x / x[p]

    for k in range(N):
        try:
            y = la.solve(A - q * np.eye(A.shape[0]), x)
        except LinAlgError:
            return q, x
        else:
            u = y[p]

            p = -1

            for i in range(len(y)):
                p += 1
                if la.norm(y, np.inf) == y[i]:
                    break

            err = la.norm(x - (y/y[p]), np.inf)
            x = y/y[p]
            if err < tol:
                u = 1/u + q
                return u, x


def deflation_wielandt(A, lam, v, x, tol = 0.0001, N = 1000):
    '''
    To approximate the second most dominant eigenvalue and an associated eigenvector of the matrix A given an
    approximation lambda to the dominant eigenvalue, an approximation v to a corresponding eigenvector, and a
    vector x of R^(n-1)
    '''
    p = -1

    for i in range(len(v)):
        p += 1
        if la.norm(v, np.inf) == v[i]:
            break

    B = np.empty((A.shape[0]-1, A.shape[0]-1))

    if i != 1:
        for k in range(i-1):
            for j in range(i-1):
                b[k][j] = A[k][j] - v[k]/v[i]*A[i][j]

    if i != 1 and i != n:
        for k in range(n-1):
            for j in range(i-1):
                b[k][j] = A[k+1][j] - v[k+1]/v[i]*A[i][j]
                b[j][k] = A[j][k+1] - v[j]/v[i]*A[i][k+1]

    if i != n:
        for k in range(n-1):
            for j in range(n-1):
                b[k][j] = A[k+1][j+1] - v[k+1]/v[i]*A[i][j+1]

    try:
        u, w = power(B, x)
    except:
        raise Exception("method failed,")

    p = np.empty((n-1, 1))
    if i != 1:
        for k in range(i-1):
            p[k] = w[k]
    p[i] = 0
    if i != n:
        for k in range(i, n):
            p[k] = w[k-1]

    y = np.empty((n, 1))

    for k in range(n):
        y[k] = (u-lam)*p[k] + (np.sum(A[i]*p[j])) * v[k]/v[i]

    return u, y




#------------------------------------------------

def test():

    A = np.array([[-2,-3],
                  [6,7]])

    B = np.array([[-4,14,0],
                  [-5,13,0],
                  [-1,0,2]])


    x = np.array([1,1])

    u, x = power(A, x, 0.0001, 1000)
    u1 = power_inverse(B, np.array([1,1,1]), 0.0001, 1000)

    print('eigenvalue of A: ', u, '\n')
    print('eigenvalue of B: ', u1)

if __name__ == '__main__':
    test()

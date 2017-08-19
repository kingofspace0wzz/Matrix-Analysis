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
        accelerate: whether to apply Aitken's acceleration method
    """
    p = -1
    u0, u1 = 0, 0
    for i in range(A.shape[1]):
        p += 1
        if la.norm(x, np.inf) == x[i]:
            break

    x = x / x[p]

    for k in range(N):
        y = np.dot(A, x)
        if accelerate == True:
            u = y[p]
            ud = u0 - (u1-u0)**2/(u-2*u1+u0)
        else:
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

        if accelerate == True:
            if err < tol and k >= 3:
                return ud, x
        else:
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



#------------------------------------------------

def test():

    A = np.array([[-2,-3],
                  [6,7]])


    x = np.array([1,1])

    u, x = power(A, x, 0.0001, 1000)

    print('eigenvalue: ', u)

if __name__ == '__main__':
    test()

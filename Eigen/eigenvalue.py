__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la

# a function that verifies Schur inequality
# test whether the sum of all square eigenValue is less than or equal to Frobenius norm of the matrix
def verSchur(A):
    '''
    a function that verifies Schur inequality
    test whether the sum of all square eigenValue is less than or equal to Frobenius norm of the matrix
    '''
    # initialize eigenValue
    eigenValue = la.eig(A)[0]

    if np.sum(eigenValue**2) <= la.norm(A, 'fro')**2:
        return True
    else:
        return False

# approximation of the upper bound of the absolute value of each eigenValue

def upperBound(A, part = ''):
    '''
    approximation of the upper bound of the absolute value of each eigenValue
    '''
    # return the approximate upper bound for the real part of all eigenValue
    if part == 'real':
        B = 1/2 * (A + A.conjugate().T)
        return B.shape[0] * abs(np.max(B))
    # return the approximate upper bound for the imaginary part of all eigenValue
    if part == 'imaginary':
        C = 1/2 * (A - A.conjugate().T)
        n = C.shape[0]
        return np.sqrt((n-1)/(2*n)) * n * abs(np.max(C))
    # return the approximate upper bound for all eigenValue
    else:
        return A.shape[0] * abs(np.max(A))

def test():

    A = np.array([[0,1,1],
                  [-1,0,1],
                  [-1,-1,0]])

    print('Upper bound of all eigenValue of A: ', upperBound(A))
    print('Upper bound of the real part all eigenValue of A: ', upperBound(A, 'real'))
    print('Upper bound of the imaginary part all eigenValue of A: ', upperBound(A, 'imaginary'))



if __name__ == '__main__':
    test()

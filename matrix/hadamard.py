__author__ = 'kingofspacewzz'

import numpy as np
from scipy import linalg as la

''' Hadamard matrix can be constructed by calling built-in function of scipy

    la.hadamard(n)


'''

# initialize a hadamard matrix by Sylvester's construction
def hadamard(n, method = 'Sylvester'):


    if method == 'Sylvester':
        H2 = np.array([[1,1],[1,-1]])
        H = np.array([[1,1],[1,-1]])
        if n == 1:
            return np.array([[1]])
        else:
            for i in range(1, n):
                H = la.kron(H2, H)
            return H


def test():

    Hard = hadamard(4)
    print(Hard)

def testIn():

    Hard = la.hadamard(16)
    print(Hard)

if __name__ == '__main__':
    test()

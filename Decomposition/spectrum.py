__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_rank as rank



def isSimple(A):

    # check if A is a squre matrix
    if A.shape[1] != A.shape[0]:
        return False

    eigenValues, eigenVectors = la.eig(A)

    '''
    # an array of all distinct eigen values with their associated algebraic multiplicity
    # keys: eigen values
    # values: algebraic multiplicity
    # {eigenValue: algebraicMulti}
    dictValues = {}
    '''
    while (eigenValues.shape[0] != 0):

        #dictValues.update({eigenValues[0]: 1})

        index = np.argwhere(abs(eigenValues - eigenValues[0]) < 0.00001)
        algebraicMulti = len(index)

        geometricMulti = eigenVectors[:, index].shape[1]

        if algebraicMulti != geometricMulti:
            return False

        #dictValues.update({eigenValues[0]: len})
        np.delete(eigenValues, index)

    # stack another spaces of eigenvalue and eigenvector

    return True


# compute the spectrum decomposition of matrix A
def spectrum_decomposition(A):

    # check if A is a simple matrix
    if isSimple(A) != True:
        raise Exception('non-simple matrix cannot be spectrum-decomposed')


    eigenValues, eigenVectors = la.eig(A)
    invVectors = la.inv(eigenVectors)
    # an array of all distinct eigen values with their associated algebraic multiplicity
    # keys: eigen values
    # values: algebraic multiplicity
    # {eigenValue: algebraicMulti}
    dictValues = {}

    while (eigenValues.shape[0] != 0):

        index = np.argwhere(abs(eigenValues - eigenValues[0]) < 0.00001)

        spectrum = eigenVectors[:, index].dot(invVectors[index, :])

        dictValues.update({eigenValues[0]: spectrum})
        np.delete(eigenValues, index)

    return dictValues


def test():

    A = np.array([[4,6,0],
                  [-3,-5,0],
                  [-3,-6,1]])

    print(spectrum_decomposition(A))

if __name__ == '__main__':
    test()

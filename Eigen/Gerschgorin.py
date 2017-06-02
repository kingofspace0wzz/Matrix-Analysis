__author__ = 'kingofspacewzz'

import numpy as np
from scipy import linalg as la

# calculate the radius of Gerschorin circle
def Gradius(A):
    R = np.empty(A.shape[0])
    for i in range(0, A.shape[0]):
        R[i] = np.sum(abs(A[i]))-abs(A[i,i])

    return R

# return a graph that shows how many connection parts consisting of Gerschorin circle A has
def Gconnection(A):

    '''
    # an array matrix that stores the distance among the central of each Gerschorin

    distance = np.zeros_like(A)
    for i in range(0, A.shape[0]):
        for j in range(i, A.shape[0]):

            if i != j:
                distance[i][j] = np.sqrt(np.squre(abs(A[i,i].real) - abs(A[j,j].real))
                                        + np.square(abs(A[i,i].imag - abs(A[j,j].imag))))
                distance[j][i] = distance[i][j]
    '''

    R = Gradius(A)
    connect = 1
    # a graph matrix that shows the relation among each Gerschorin circle
    graph = np.zeros((A.shape[0], A.shape[0]))
    # construct the graph that shows the relation among each Gerschorin circle
    for i in range(0, A.shape[0]):
        for j in range(i+1, A.shape[0]):

            distance = np.sqrt(np.square(abs(A[i,i].real) - abs(A[j,j].real))
                                    + np.square(abs(A[i,i].imag - abs(A[j,j].imag))))
            if distance < R[i] + R[j] :
                graph[i][j] = 1
                graph[j][i] = 1

    return graph


def test():

    A = np.array([[1j,0.1,0.2,0.3],
                  [0.5,3,0.1,0.2],
                  [1,0.3,1,0.5],
                  [0.2,-0.3,-0.1,-4]])

    print('Gradius of A: ', Gradius(A))
    print('Connection: ', Gconnection(A))

if __name__ == '__main__':
    test()

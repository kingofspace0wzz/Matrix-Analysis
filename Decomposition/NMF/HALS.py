__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_rank as rank

# solve NMF by hierarchical alternating least squares, returns the approximation matrices and the residue
def HALS(A, k, epsilon = 0.00001):

    W = np.random.random_sample((A.shape[0], k))
    H = np.random.random_sample((A.shape[1], k))

    proGradientW = 2 * W.dot(H.T).dot(H) - 2 * A.dot(H)
    proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T.dot(W)

    for i in range(proGradientW.shape[0]):
        for j in range(proGradientW.shape[1]):

            if proGradientW[i, j] >= 0 | W[i, j] <= 0:
                proGradientW[i, j] = 0

    for i in range(proGradientH.shape[0]):
        for j in range(proGradientH.shape[1]):

            if proGradientH[i, j] >= 0 | H[i, j] <= 0:
                proGradientH[i, j] = 0

    initalDelta = np.sqrt( np.squre(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

    while True:

        for i in range(k):
            temp = W[:, i].reshape((A.shape[0], 1))
            temp = W[:, i].reshape((A.shape[0], 1))  + ( (A.dot(H))[:, i].reshape((A.shape[0], 1)) - (W.dot(H.T).dot(H))[:, i].reshape((A.shape[0], 1)) ) / (H.T.dot(H)[i, i])
            W[:, i] = temp.reshape((1, A.shape[0]))

            for j in range(len(W[:, i])):
                if W[:, i][j] < 0:
                    W[:, i][j] = 0

        for i in range(k):
            temp = H[:, i].reshape((A.shape[1], 1))
            temp = H[:, i].reshape((A.shape[1], 1)) + ( (A.T.dot(W))[:, i].reshape((A.shape[1], 1)) - (H.dot(W.T.dot(W)))[:, i].reshape((A.shape[1], 1)) ) / W.T.dot(W)[i, i]
            H[:, i] = temp.reshape((1, A.shape[1]))

            for j in range(len(H[:, i])):
                if H[:, i][j] < 0:
                    H[:, i][j] = 0

        proGradientW = 2 * W.dot(H.T).dot(H) - 2 * A.dot(H)
        proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T.dot(W)

        for i in range(proGradientW.shape[0]):
            for j in range(proGradientW.shape[1]):

                if proGradientW[i, j] >= 0 or W[i, j] <= 0:
                    proGradientW[i, j] = 0

        for i in range(proGradientH.shape[0]):
            for j in range(proGradientH.shape[1]):

                if proGradientH[i, j] >= 0 or H[i, j] <= 0:
                    proGradientH[i, j] = 0

        delta = np.sqrt( np.square(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

        if delta / initalDelta <= epsilon:
            break

    residue = la.norm(A - W.dot(H.T), 2)

    return W, H, residue



# regularized HALS method, keeping the results small and sparse
def regularized_HALS(A, k, epsilon = 0.00001, alpha = 0.01, beta = 0.01):

    W = np.random.random_sample((A.shape[0], k))
    H = np.random.random_sample((A.shape[1], k))

    proGradientW = 2 * W.dot(H.T).dot(H) - 2 * A.dot(H)
    proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T.dot(W)

    for i in range(proGradientW.shape[0]):
        for j in range(proGradientW.shape[1]):

            if proGradientW[i, j] >= 0 or W[i, j] <= 0:
                proGradientW[i, j] = 0

    for i in range(proGradientH.shape[0]):
        for j in range(proGradientH.shape[1]):

            if proGradientH[i, j] >= 0 or H[i, j] <= 0:
                proGradientH[i, j] = 0

    initalDelta = np.sqrt( np.square(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

    while True:

        for i in range(k):
            temp = W[:, i].reshape((A.shape[0], 1))
            temp = W[:, i].reshape((A.shape[0], 1)) * H.T.dot(H)[i, i] / (H.T.dot(H)[i, i] + alpha) + ( (A.dot(H))[:, i].reshape((A.shape[0], 1)) - (W.dot(H.T).dot(H))[:, i].reshape((A.shape[0], 1)) ) / (H.T.dot(H)[i, i] + alpha)
            W[:, i] = temp.reshape((1, A.shape[0]))

            for j in range(len(W[:, i])):
                if W[:, i][j] < 0:
                    W[:, i][j] = 0

        for i in range(k):
            temp = H[:, i].reshape((A.shape[1], 1))
            temp = H[:, i].reshape((A.shape[1], 1)) + ( (A.T.dot(W))[:, i].reshape((A.shape[1], 1)) - (H.dot(W.T.dot(W) + beta))[:, i].reshape((A.shape[1], 1)) ) / (W.T.dot(W)[i, i] + beta)
            H[:, i] = temp.reshape((1, A.shape[1]))

            for j in range(len(H[:, i])):
                if H[:, i][j] < 0:
                    H[:, i][j] = 0

        proGradientW = 2 * W.dot(H.T).dot(H) - 2 * A.dot(H)
        proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T.dot(W)

        for i in range(proGradientW.shape[0]):
            for j in range(proGradientW.shape[1]):

                if proGradientW[i, j] >= 0 or W[i, j] <= 0:
                    proGradientW[i, j] = 0

        for i in range(proGradientH.shape[0]):
            for j in range(proGradientH.shape[1]):

                if proGradientH[i, j] >= 0 or H[i, j] <= 0:
                    proGradientH[i, j] = 0

        delta = np.sqrt( np.square(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

        if delta / initalDelta <= epsilon:
            break

    residue = la.norm(A - W.dot(H.T), 2)

    return W, H, residue


def test():

    A = np.array([[1.2,2.1],
                  [2.9,4.3],
                  [5.2,6.1],
                  [6.8,8.1]])

    W, H, residue = regularized_HALS(A, 2)

    print(W, '\n', H, '\n', residue)

if __name__ == '__main__':
    test()

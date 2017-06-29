__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_rank as rank

# solve NMF by hierarchical alternating least squares, returns the approximation matrices and the residue
def HALS(A, k, epsilon = 0.00001):

    W = np.random.random_sample((A.shape[0], k))
    H = np.random.random_sample((A.shape[1], k))

    proGradientW = 2 * H.dot(H.T).dot(H) - 2 * A * H
    proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T * W

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
            W[:, i].reshape((A.shape[0], 1)) = W[:, i].reshape((A.shape[0], 1)) +
                ( (A.dot(H))[:, i].reshape((A.shape[0], 1)) - (W.dot(H.T).dot(H))[:, i].reshape((A.shape[0], 1)) )
                / (H.T.dot(H)[i, i])

            for j in range(len(W[:, i])):
                if W[:, i][j] < 0:
                    W[:, i][j] = 0

        for i in range(k):
            H[:, i].reshape((A.shape[1], 1)) = H[:, i].reshape((A.shape[1], 1)) +
                ( (A.T.dot(W))[:, i].reshape((A.shape[1], 1)) - (H.dot(W.T).dot(W))[:, i].reshape((A.shape[1], 1)) )
                / (W.T.dot(W)[i, i])

            for j in range(len(H[:, i])):
                if H[:, i][j] < 0:
                    H[:, i][j] = 0

        proGradientW = 2 * H.dot(H.T).dot(H) - 2 * A * H
        proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T * W

        for i in range(proGradientW.shape[0]):
            for j in range(proGradientW.shape[1]):

                if proGradientW[i, j] >= 0 | W[i, j] <= 0:
                    proGradientW[i, j] = 0

        for i in range(proGradientH.shape[0]):
            for j in range(proGradientH.shape[1]):

                if proGradientH[i, j] >= 0 | H[i, j] <= 0:
                    proGradientH[i, j] = 0

        delta = np.sqrt( np.squre(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

        if delta / initalDelta <= epsilon:
            break

    residue = la.norm(A - W.dot(H.T), 2)

    return W, H, residue



# regularized HALS method, keeping the results small and sparse
def regularized_HALS(A, k, epsilon = 0.00001, alpha = 0.01, beta = 0.01):

    W = np.random.random_sample((A.shape[0], k))
    H = np.random.random_sample((A.shape[1], k))

    proGradientW = 2 * H.dot(H.T).dot(H) - 2 * A * H
    proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T * W

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
            W[:, i].reshape((A.shape[0], 1)) = W[:, i].reshape((A.shape[0], 1)) * H.T.dot(H)[i, i] / (H.T.dot(H)[i, i] + alpha) +
                ( (A.dot(H))[:, i].reshape((A.shape[0], 1)) - (W.dot(H.T).dot(H))[:, i].reshape((A.shape[0], 1)) )
                / (H.T.dot(H)[i, i] + alpha)

            for j in range(len(W[:, i])):
                if W[:, i][j] < 0:
                    W[:, i][j] = 0

        for i in range(k):
            H[:, i].reshape((A.shape[1], 1)) = H[:, i].reshape((A.shape[1], 1)) +
                ( (A.T.dot(W))[:, i].reshape((A.shape[1], 1)) - (H.dot(W.T).dot(W))[:, i].reshape((A.shape[1], 1)) - H.dot(np.ones((k, k))) * beta )
                / (W.T.dot(W)[i, i] + beta)

            for j in range(len(H[:, i])):
                if H[:, i][j] < 0:
                    H[:, i][j] = 0

        proGradientW = 2 * H.dot(H.T).dot(H) - 2 * A * H
        proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T * W

        for i in range(proGradientW.shape[0]):
            for j in range(proGradientW.shape[1]):

                if proGradientW[i, j] >= 0 | W[i, j] <= 0:
                    proGradientW[i, j] = 0

        for i in range(proGradientH.shape[0]):
            for j in range(proGradientH.shape[1]):

                if proGradientH[i, j] >= 0 | H[i, j] <= 0:
                    proGradientH[i, j] = 0

        delta = np.sqrt( np.squre(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

        if delta / initalDelta <= epsilon:
            break

    residue = la.norm(A - W.dot(H.T), 2)

    return W, H, residue

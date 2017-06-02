__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la


# return a scaled matrix of A
def scaleMatrix(A, scale = [1,1]):

    sm = int(A.shape[0] / scale[0])
    sn = int(A.shape[1] / scale[1])

    output = np.empty((sm, sn))

    for i in range(sm):
        for j in range(sn):
            sum = 0
            for si in range(i * scale[0], (i+1) * scale[0]):
                for sj in range(j * scale[1], (j+1) * scale[1]):
                    sum += A[si, sj]

            output[i,j] = sum / (scale[0] * scale[1])

    return output

# rotate matrix by 180 degrees
def rot180(A):



    return rot90(rot90(A))

# rotate matrix by 90 degrees
def rot90(A):


    m = A.shape[0]
    n = A.shape[1]
    output = np.empty((n, m))

    for i in range(m):
        for j in range(n):

            output[j, m-1-i] = A[i, j]

    A = output
    return A

# rotation, counterclock
def Crot90(A):

    return rot90(rot180(A))

# return the Full convolution between Matrix A and the kernel
def conFull(A, kernel):

    kernel = rot180(kernel)

    if len(A.shape) != len(kernel.shape):
        raise Exception('Kernel is larger than the matrix', '\n')

    if len(A.shape) == 2:

        m = A.shape[0]
        n = A.shape[1]
        km = kernel.shape[0]
        kn = kernel.shape[1]

        extendMatrix = np.empty((m+2*(km-1), n+2*(kn-1)))
        for i in range(m):
            for j in range(n):
                extendMatrix[i+km-1, j+kn-1] = A[i, j]
        return conValid(extendMatrix, kernel)

    if len(A.shape) == 3:

        m = A.shape[0]
        n = A.shape[1]
        h = A.shape[2]
        km = kernel.shape[0]
        kn = kernel.shape[1]
        kh = kernel.shape[2]

        extendMatrix = np.ones((m+2*(km-1), n+2*(kn-1), h+2*(kh-1)))
        for i in range(m):
            for j in range(n):
                for k in range(h):
                    extendMatrix[i+km-1, j+kn-1, k+kh-1] = A[i, j, k]
        return conValid(extendMatrix, kernel)

    else:
        raise Exception("Under Construction")


# return the valid convolution between Matrix A and the kernel
def conValid(A, kernel):

    kernel = rot180(kernel)

    if len(A.shape) != len(kernel.shape):
        raise Exception("Kernel is large than the matrix", '\n')

    if len(A.shape) == 2:

        m = A.shape[0]
        n = A.shape[1]
        km = kernel.shape[0]
        kn = kernel.shape[1]

        kms = m - km + 1
        kns = n - kn + 1

        output = np.empty((kms, kns))

        for i in range(kms):
            for j in range(kns):
                sum = 0
                for ki in range(km):
                    for kj in range(kn):
                        sum += A[i+ki, j+kj] * kernel[ki, kj]

                output[i, j] = sum

        return output

    if len(A.shape) == 3:

        m = A.shape[0]
        n = A.shape[1]
        h = A.shape[2]
        km = kernel.shape[0]
        kn = kernel.shape[1]
        kh = kernel.shape[2]

        kms = m - km + 1
        kns = n - kn + 1
        khs = h - kh + 1

        output = np.empty((kms, kns, khs))
        for i in range(kms):
            for j in range(kns):
                for k in range(khs):
                    sum = 0
                    for ki in range(km):
                        for kj in range(kn):
                            for kk in range(kh):
                                sum += A[i+ki, j+kj, k+kh] * kernel[ki, kj, kk]


                    output[i, j, k] = sum

        return output

    if len(A.shape) == 1:
        return np.convolve(A, kernel)

    else:
        raise Exception('Under Construction')

        length = len(A.shape)

        a = np.empty((length))
        for i in range(length):
            a[i] = A.shape[i]

        b = np.empty((length))
        for i in range(length):
            b[i] = kernel.shape[i]

        c = np.empty((length))
        for i in range(length):
            c[i] = a[i] - b[i] + 1

        output = np.empty(c)



#--------------------------------------------------------------

# test whether scaleMatrix() works
def testScale():

    A = np.array([[1,3,5,7,9,10],
                  [2,4,6,8,10,12],
                  [3,5,6,8,11,13],
                  [5,7,9,11,13,15],
                  [1,2,3,4,5,6],
                  [2,3,4,5,6,7]])

    print('A: ', A, '\n', 'Scale of A: ', scaleMatrix(A, (2,2)))

def test():

    A = np.array([[1,3,5,7,9,10],
                  [2,4,6,8,10,12],
                  [3,5,6,8,11,13],
                  [5,7,9,11,13,15],
                  [1,2,3,4,5,6],
                  [2,3,4,5,6,7]])

    kernel = np.array([[1,2,1],
                       [2,1,2]])

    print('Rotate A: ', rot90(kernel))
    print("Convolution: ", conValid(A, kernel))
    print('Full Convolution: ', conFull(A, kernel))


if __name__ == '__main__':

    test()

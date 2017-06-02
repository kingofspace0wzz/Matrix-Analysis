
import time
import numpy as np
from scipy import linalg as la
from functools import wraps

def timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
        (function.func_name, str(t1-t0)))
        return result
    return function_timer


def test():
    @timer
    def calInverse(A):
        inverse = la.inv(A)

    A = np.array([[1,4,6],
                  [4,5,7],
                  [6,7,9]])

    calInverse(A)

if __name__ == '__main__':
    test()

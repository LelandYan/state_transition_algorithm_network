# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 22:05'

import numpy as np
from binary_state_transition_algorithm.op_get_diffnumber import op_get_diffnumber
from binary_state_transition_algorithm.op_getBest import op_getBest
def op_symmetry(oldBest, m, n):
    '''
       :param oldBest:the matrix shape is (m*n,1)
       :param m: the number of sample
       :param n: the number of feature
       :return:the swapped matrix
       '''
    size = m * n
    lens = len(oldBest)
    oldBest = oldBest.reshape(lens, 1)
    oldBest = np.tile(oldBest, (m, 1))
    coefficient = np.eye(size, size)
    for i in range(m):
        # get the different number(a,b)
        a, b = op_get_diffnumber(i * n, i * n + n)
        s = coefficient[a:b, :]
        coefficient[a:b, :] = s[::-1,:]
    newBest = coefficient.dot(oldBest)
    return op_getBest(newBest, m, n)

if __name__ == '__main__':
    oldBest = np.array([1, 1, 0, 1, 0, 1])
    print(op_symmetry(oldBest, 1, 6))
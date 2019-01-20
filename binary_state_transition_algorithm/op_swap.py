# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 16:57'

import numpy as np
from binary_state_transition_algorithm.op_getBest import op_getBest
from binary_state_transition_algorithm.op_get_diffnumber import op_get_diffnumber

def op_swap(oldBest, m, n):
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
        # get the different number
        a,b = op_get_diffnumber(i * n,i * n + n)
        coefficient[[a, b], :] = coefficient[[b, a], :]
    newBest = coefficient.dot(oldBest)
    return op_getBest(newBest,m,n)



if __name__ == '__main__':
    oldBest = np.array([1, 1, 0, 1, 0, 1])
    print(op_swap(oldBest, 1, 6))

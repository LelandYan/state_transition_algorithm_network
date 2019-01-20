# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 21:26'

import numpy as np
import random as rd
from binary_state_transition_algorithm.op_getBest import op_getBest
from binary_state_transition_algorithm.op_get_diffnumber import op_get_diffnumber


def op_shift(oldBest, m, n):
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
        a, b ,c= op_get_diffnumber(i * n, i * n + n,3)
        coefficient[[a, b, c], :] = coefficient[[b, c, a], :]
    newBest = coefficient.dot(oldBest)
    return op_getBest(newBest, m, n)


if __name__ == '__main__':
    oldBest = np.array([1, 1, 0, 1, 0, 1])
    print(op_shift(oldBest, 1, 6))
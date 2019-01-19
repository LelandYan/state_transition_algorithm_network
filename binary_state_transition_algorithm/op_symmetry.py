# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 22:05'

import numpy as np
import random as rd


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
        s = rd.sample(range(i * n, i * n + n), 2)
        a = s[0]
        b = s[1]
        if a > b:
            a,b = b,a
        s = coefficient[a:b, :]
        coefficient[a:b, :] = s[::-1,:]
    return (coefficient.dot(oldBest))

if __name__ == '__main__':
    oldBest = np.array([1, 1, 0, 1, 0, 1])
    print(op_symmetry(oldBest, 1, 6))
# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 16:57'

import numpy as np
import random as rd

def op_swap(oldBest,m,n):
    '''
    :param oldBest:the matrix shape is (m*n,1)
    :param m: the number of sample
    :param n: the number of feature
    :return:the swapped matrix
    '''
    size = m * n
    lens = len(oldBest)
    oldBest = oldBest.reshape(lens,1)
    oldBest = np.tile(oldBest,(m,1))
    coefficient = np.eye(size,size)
    for i in range(m):
        a,b = rd.randint(i*n,i*n+n-1),rd.randint(i*n,i*n+n-1)
        coefficient[[a,b],:] = coefficient[[b,a],:]
    return (coefficient.dot(oldBest))


if __name__ == '__main__':
    oldBest = np.array([1,1,0,1,0,1])
    print(op_swap(oldBest,2000,6).shape)
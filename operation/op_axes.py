# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 8:48'

import numpy as np
import random as rd

def op_axes(Best,SE,delta):
    n = Best.size
    A = np.zeros((n,SE))
    index = np.random.randint(0,n,(1,SE))
    A[index,list(range(SE))] = 1
    Best = Best.reshape(n,1)
    a = np.tile(Best,SE)
    b = np.array([rd.gauss(0, 1) for _ in range(n * SE)]).reshape(n, SE)
    c = delta * b * A * a
    y = a + c
    y = y.transpose()
    return y

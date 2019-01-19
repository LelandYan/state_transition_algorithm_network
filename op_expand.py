# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 8:40'

import numpy as np
import random as rd


def op_expand(Best, SE, gamma):
    n = Best.size
    Best = Best.reshape(n, 1)
    a = np.tile(Best, SE)
    b = np.array([rd.gauss(0,1) for _ in range(n*SE)]).reshape(n,SE)
    y = a + gamma * b * a
    y = y.transpose()
    return y



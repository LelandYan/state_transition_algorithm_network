# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 8:24'

import numpy as np

def op_translate(oldBest,newBest,SE,beta):
    '''
    :param oldBest:原点
    :param newBest:新点
    :param SE:个体数量
    :param beta:搜索的最大长度
    :return:
    '''
    n = oldBest.size
    oldBest = oldBest.reshape(n,1)
    newBest = newBest.reshape(n,1)
    diff = newBest - oldBest
    a = np.tile(newBest,SE)
    b = beta/(np.linalg.norm(diff) + 2e-16)
    c = np.tile(np.random.uniform(0,1,(1,SE)),n).reshape(n,SE)*np.tile(diff,SE)
    y = a + b * c
    y = y.transpose()
    return y
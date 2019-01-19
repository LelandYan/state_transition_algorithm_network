# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 8:07'
import numpy as np


def op_rotate(Best, SE, alpha):
    '''
    :param Best:以列表的形式传入特征
    :param SE: 个体的数量
    :param alpha: 旋转变换具有在半径为alpha的超球体内搜索
    :return: 进行完rotate后的SE个样本
    '''
    n = Best.size
    # 改变矩阵的形状
    Best = Best.reshape(n, 1)
    # 矩阵的形式变成 特征数目x样本数量
    a = np.tile(Best, SE)
    b = np.dot(np.random.uniform(-1, 1, (SE * n, n)), Best).reshape(n, SE)
    # 这里加入2e-16是为了防止除数为0
    c = 1.0 / n / (np.linalg.norm(Best) + 2e-16)
    y = a + alpha * c * b
    y = y.transpose()
    return y




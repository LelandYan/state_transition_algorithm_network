# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/20 13:22'
import numpy as np
import random as rd
from binary_state_transition_algorithm.fitness import fitness


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
        a, b, c = op_get_diffnumber(i * n, i * n + n, 3)
        coefficient[[a, b, c], :] = coefficient[[b, c, a], :]
    newBest = coefficient.dot(oldBest)
    return op_getBest(newBest, m, n)


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
        coefficient[a:b, :] = s[::-1, :]
    newBest = coefficient.dot(oldBest)
    return op_getBest(newBest, m, n)


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
        a, b = op_get_diffnumber(i * n, i * n + n)
        coefficient[[a, b], :] = coefficient[[b, a], :]
    newBest = coefficient.dot(oldBest)
    return op_getBest(newBest, m, n)


def op_substitute(oldBest, m, n):
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
        a = rd.randint(i * n, i * n + n - 1)
        coefficient[a, a] = 0
    newBest = coefficient.dot(oldBest)
    return op_getBest(newBest, m, n)


def op_get_diffnumber(a, b, n=2):
    s = rd.sample(range(a, b), n)
    s.sort()
    return s


def op_getBest(newBest, m, n):
    features = []
    accuracy = []
    for i in range(m):
        features.append(newBest[i * n:i * n + n, :].sum())
        accuracy.append(fitness(newBest[i * n:i * n + n, :]))
    index = accuracy.index(max(accuracy))
    return newBest[index * n:index * n + n, :], accuracy[index], features[index]

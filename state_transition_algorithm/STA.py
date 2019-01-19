# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 11:18'

from rotate import *
from expand import *
from axesion import *

def STA(func,Best,SE,Range,Iterations):
    alpha_max = 1
    alpha_min = 1e-4
    alpha = alpha_max
    beta = 1
    gamma = 1
    delta = 1
    fc = 2
    history = np.empty((Iterations,1))
    fBest = func(Best[0])

    for iter in range(Iterations):
        Best, fBest = expand(func, Best, fBest, SE, Range, beta, gamma)
        Best, fBest = rotate(func, Best, fBest, SE, Range, alpha, beta)
        Best, fBest = axesion(func, Best, fBest, SE, Range, beta, delta)
        history[iter] = fBest
        alpha = alpha / fc if alpha > alpha_min else alpha_max

    return Best,fBest,history
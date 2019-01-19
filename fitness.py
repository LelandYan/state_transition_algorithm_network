# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 9:45'
import numpy as np



def fitness(func, State):
    # SE = State.shape[0]
    fState = list(map(func, State))
    fGBest = np.min(fState)
    Best = State[fState.index(fGBest)]
    return Best, fGBest

# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 10:00'

import numpy as np
from op_translation import op_translate
import random as rd
from fitness import fitness

def op_expand(Best, SE, gamma):
    n = Best.size
    Best = Best.reshape(n, 1)
    a = np.tile(Best, SE)
    b = np.array([rd.gauss(0,1) for _ in range(n*SE)]).reshape(n,SE)
    y = a + gamma * b * a
    y = y.transpose()
    return y

def expand(func,Best,fBest,SE,Range,beta,gamma):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    oldBest = Best
    State = op_expand(Best, SE, gamma)
    changeRows = State > Pop_Ub
    State[changeRows] = Pop_Ub[changeRows]
    changeRows = State < Pop_Lb
    State[changeRows] = Pop_Lb[changeRows]
    newBest, fGBest = fitness(func, State)
    if fGBest < fBest:
        fBest, Best = fGBest, newBest
        State = op_translate(oldBest, Best, SE, beta)
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        newBest, fGBest = fitness(func, State)
        if fGBest < fBest:
            fBest, Best = fGBest, newBest
    return Best, fBest
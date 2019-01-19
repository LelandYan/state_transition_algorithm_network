# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 10:14'

import numpy as np
import random as rd
from op_translation import op_translate
from fitness import fitness

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
def axesion(func,Best,fBest,SE,Range,beta,delta):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    oldBest = Best
    State = op_axes(Best, SE, delta)
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
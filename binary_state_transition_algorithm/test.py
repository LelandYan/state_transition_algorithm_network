# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 17:45'

import numpy as np
import random as rd

a = np.array([[1,2,3],[4,5,6]])

s1 = a[:,::-1][1,0:2]
a[0,0:2] = a[:,::-1][0,0:2]
#a[0:2, :] = s1
# s2 = str(s1)
# print(s2)
# b = {s2:1}
# print(b)
# print(s2)
a = a.reshape(6,1)
print(a.flatten())
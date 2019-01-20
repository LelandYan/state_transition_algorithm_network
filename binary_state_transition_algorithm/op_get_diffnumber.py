# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/20 11:03'


import random as rd

def op_get_diffnumber(a,b,n=2):
    s = rd.sample(range(a, b), n)
    s.sort()
    return s

if __name__ == '__main__':
    a,b = op_get_diffnumber(1,3)
    print(a,b)
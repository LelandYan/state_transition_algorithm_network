# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 11:45'

from STA import *
from functools import reduce
from operator import mul
import math
import matplotlib.pyplot as plt
def Griewank(s):
    t1 = sum(map(lambda x: 1 / 4000 * x ** 2, s))
    n = len(s)
    t2 = map(lambda x, y: math.cos(x / np.sqrt(y)), s, range(1, n + 1))
    t3 = reduce(mul, t2)
    return t1 - t3 + 1

# 个体的数量
SE = 30
# 特征的数量
Dim = 5
# 限制的上下界限
Range = np.tile([[-30],[30]],Dim)
# 迭代的次数
Iterations = 500
# 初始化数据
Best0 = np.array(Range[0,:] + (Range[1,:]-Range[0,:]*np.random.uniform(0,1,(1,Dim))))
# STA算法进行训练
xmin,fxmin,history = STA(Griewank,Best0,SE,Range,Iterations)
print("此函数最小值点:",xmin,'\n',"此函数最小值:",fxmin)
plt.plot(history,'b.-')
plt.semilogy(history,'b.-') # 对数曲线
plt.xlabel('Iterations')
plt.ylabel('fitness')
plt.show()
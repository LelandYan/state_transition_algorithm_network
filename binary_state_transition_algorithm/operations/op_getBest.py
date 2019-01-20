# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/20 11:33'


from binary_state_transition_algorithm.fitness import fitness

def op_getBest(newBest,m,n):
    features = []
    accuracy = []
    for i in range(m):
        features.append(newBest[i * n:i * n + n, :].sum())
        accuracy.append(fitness(newBest[m * n:m * n + n, :]))
    index = accuracy.index(max(accuracy))
    return newBest[index * n:index * n + n, :],accuracy[index],features[index]
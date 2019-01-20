# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/20 10:09'

from binary_state_transition_algorithm import model
import pandas as pd

# the path and name of file
CSV_FILE_PATH = 'csv_result-ALL-AML_train.csv'
# read the file
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
# the eigenvalue of file
input_data = df.values[:, 1:shapes[1] - 1]
# the result of file
result = df.values[:, shapes[1] - 1:shapes[1]]


def translate(pop):
    index_list = []
    for i in range(len(pop)):
        if pop[i] == 1:
            index_list.append(i)
    return index_list

def fitness(Best):
    newBest = translate(Best.flatten())
    data = input_data[:,newBest]
    return model.Neural_Network().__int__(data, result)[0]

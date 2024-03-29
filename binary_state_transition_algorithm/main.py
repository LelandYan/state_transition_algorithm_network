# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/19 23:13'

from binary_state_transition_algorithm.operation import *
import pandas as pd
import random as rd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn', lineno=196)
N_GENERATIONS = 100
POP_SIZE = 50


def b_sta():
    CSV_FILE_PATH = 'parkinsons.csv'
    # read the file
    df = pd.read_csv(CSV_FILE_PATH)
    shapes = df.values.shape
    # the eigenvalue of file
    input_data = df.values[:, 1:shapes[1] - 1]
    # the result of file
    value_len = input_data.shape[1]
    one = []
    for i in range(value_len):
        one.append(rd.randint(0, 1))
    begin_one = np.array(one)
    new_one = begin_one
    accuracy = 0
    feature = 0
    for i in range(N_GENERATIONS):
        new_one, accuracy, feature = op_swap(new_one, POP_SIZE, value_len)
        new_one, accuracy, feature = op_shift(new_one, POP_SIZE, value_len)
        new_one, accuracy, feature = op_symmetry(new_one, POP_SIZE, value_len)
        new_one, accuracy, feature = op_substitute(new_one, POP_SIZE, value_len)
    print("acc:", accuracy, "  ", "the number of features:", feature)


if __name__ == '__main__':
    b_sta()

import numpy as np
import random
import tensorflow as tf

dataset_file = "./dataset.txt"
dataset_line_num = 1e5
test_dataset_num = 1e2
data_range = int(1e5)


def gen_dataset():
    res = []
    n = int(1e5)
    for i in range(0, n):
        value = random.randint(-data_range, data_range)
        label = 0 if value < 0 else 1
        res.append("{} {}\n".format(value, label))
    with open(dataset_file, "w") as f:
        f.writelines(res)


def load_dataset():
    x_train, y_train = [], []
    x_test, y_test = [], []
    with open(dataset_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            value, label = line.split(" ")
            if i < (dataset_line_num - test_dataset_num):
                x_train.append(int(value))
                y_train.append(int(label))
            else:
                x_test.append(int(value))
                y_test.append(int(label))
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    gen_dataset()

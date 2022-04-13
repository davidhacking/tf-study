# encoding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

"""
z = (x**2+y-11)**2+(x+y**-7)**2
"""


# 这里使用1维主要是后续tf优化
def himmelblau(t):
    x = t[0]
    y = t[1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


learning_rate = 1e-3


def solve_min():
    x = tf.Variable([1.0, 0.0]) # 必须使用float类型，不然无法进行GradientTape watch
    for step in range(400):

        with tf.GradientTape() as tape:
            y = himmelblau(x)

        grads = tape.gradient(y, [x])
        if grads[0] is None:
            continue
        x.assign_sub(learning_rate * grads[0])
        if step % 10 == 0:
            print("step={}, x={}, y={}".format(step, x, y))


if __name__ == "__main__":
    # f = plt.figure("himmelblau")
    # axis = f.gca(projection='3d')
    # x, y = np.meshgrid(np.arange(-100, 100, .1), np.arange(-100, 100, .1))
    # axis.plot_surface(x, y, himmelblau([x, y]))
    # plt.show()
    solve_min()

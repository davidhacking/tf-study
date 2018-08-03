# encoding=utf-8

import numpy as np


def softmax(x):
	"""Compute the softmax of vector x."""
	exp_x = np.exp(x)
	softmax_x = exp_x / np.sum(exp_x)
	return softmax_x


def reduce_sum(a):
	return np.sum(a, axis=None)


def iterate():
	pass


if __name__ == "__main__":
	import tensorflow.examples.tutorials.mnist.input_data as input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	W = np.zeros((784, 10))
	b = np.zeros(10)
	for i in range(mnist.train.num_examples):
		batch_xs, batch_ys = mnist.train.next_batch(1)
		y = softmax(np.dot(batch_xs, W) + b)
		loss = -reduce_sum(batch_ys * np.log(y))
	pass

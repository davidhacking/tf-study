# cofing=utf-8
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
	import tensorflow.examples.tutorials.mnist.input_data as input_data

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	for i in range(mnist.train.num_examples):
		batch_xs, batch_ys = mnist.train.next_batch(1)

		img = Image.fromarray(batch_xs.reshape(28, 28) * 255)
		imgplot = plt.imshow(img)
		plt.show()
	pass

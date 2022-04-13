import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# https://www.youtube.com/watch?v=CCuwdCgg56E&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=52
#
# (x_train, y_train), _ = datasets.mnist.load_data()
# x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
#
# print(x_train.shape, y_train.shape)
# print(tf.reduce_min(x_train), tf.reduce_max(x_train))
# print(tf.reduce_min(y_train), tf.reduce_max(y_train))
#
# train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
#
# w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.01))
# b1 = tf.Variable(tf.zeros([256]))
# w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.01))
# b2 = tf.Variable(tf.zeros([128]))
# w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.01))
# b3 = tf.Variable(tf.zeros([10]))
#
# learning_rate = 0.001
#
# for epoch in range(10):
#     for i, (x, y) in enumerate(train_db):
#         x = tf.reshape(x, [-1, x_train.shape[1] * x_train.shape[2]])
#
#         with tf.GradientTape() as tape:
#             h1 = tf.nn.relu(x @ w1 + b1)
#             h2 = tf.nn.relu(h1 @ w2 + b2)
#             out = h2 @ w3 + b3
#             y_one_hot = tf.one_hot(y, depth=10)
#             loss = tf.reduce_mean(tf.square(y_one_hot - out))
#         grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
#
#         w1.assign_sub(learning_rate * grads[0])
#         b1.assign_sub(learning_rate * grads[1])
#         w2.assign_sub(learning_rate * grads[2])
#         b2.assign_sub(learning_rate * grads[3])
#         w3.assign_sub(learning_rate * grads[4])
#         b3.assign_sub(learning_rate * grads[5])
#         if i % 100 == 0:
#             print(epoch, i, "loss: ", float(loss))

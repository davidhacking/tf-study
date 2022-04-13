import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
import cafar10db

"""
https://www.youtube.com/watch?v=ZB2pDS4sUhI&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=100
"""

# 5层正好把32维降到1维
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
]


def main():
    train_db, test_db = cafar10db.load_db()
    conv_net = Sequential(conv_layers)
    conv_net.build(input_shape=[None, 32, 32, 3])
    # test
    # print(conv_net(tf.random.normal([4, 32, 32, 3])).shape)
    dense_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=tf.nn.relu),
    ])
    dense_net.build(input_shape=[None, 512])
    opt = optimizers.Adam(learning_rate=1e-4)
    variables = conv_net.trainable_variables + dense_net.trainable_variables
    for epoch in range(50):
        for step, (x, y) in enumerate(train_db):
            grads, loss = gradient(conv_net, dense_net, x, y, variables)
            opt.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                acc = test(test_db, conv_net, dense_net)
                print("epoch={}, step={}, loss={}, acc={}".format(epoch, step, float(loss), acc))


def model(conv_net, dense_net, x):
    return dense_net(tf.reshape(conv_net(x), [-1, 512]))


def test(test_db, conv_net, dense_net):
    total, correct = 0, 0
    for x, y in test_db:
        logits = model(conv_net, dense_net, x)
        pred = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        y = tf.cast(tf.argmax(y, axis=1), dtype=tf.int32)
        equals = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
        correct += int(equals)
        total += x.shape[0]
    return correct / total


def gradient(conv_net, dense_net, x, y, variables):
    with tf.GradientTape() as tape:
        logits = model(conv_net, dense_net, x)
        loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    return tape.gradient(loss, variables), loss


if __name__ == '__main__':
    main()

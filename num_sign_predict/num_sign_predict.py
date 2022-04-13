import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import num_dataset

"""
为啥 w.shape = [1, 2] b.shape = [2]就不行呢？表达能力不够？
"""


def network(x, w1, b1, w2, b2, w3, b3):
    h1 = tf.nn.relu(x @ w1 + b1)
    h2 = tf.nn.relu(h1 @ w2 + b2)
    out = tf.sigmoid(h2 @ w3 + b3)
    return out


def train(x_train, y_train):
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)

    w1 = tf.Variable(tf.random.truncated_normal([1, 256], stddev=0.01))
    b1 = tf.Variable(tf.zeros([256]))
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.01))
    b2 = tf.Variable(tf.zeros([128]))
    w3 = tf.Variable(tf.random.truncated_normal([128, 2], stddev=0.01))
    b3 = tf.Variable(tf.zeros([2]))

    learning_rate = 0.001

    for i, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 1])
        with tf.GradientTape() as tape:
            out = network(x, w1, b1, w2, b2, w3, b3)
            y_one_hot = tf.one_hot(y, depth=2)
            loss = tf.reduce_mean(tf.square(y_one_hot - out))
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        w1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        w2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
        w3.assign_sub(learning_rate * grads[4])
        b3.assign_sub(learning_rate * grads[5])
        if i % 100 == 0:
            print(i, "loss: ", float(loss))
    return w1, b1, w2, b2, w3, b3


def test(x_test, y_test, args):
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
    w1, b1, w2, b2, w3, b3 = args
    correct, total = 0, 0
    for i, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 1])
        out = network(x, w1, b1, w2, b2, w3, b3)
        probability = tf.nn.softmax(out, axis=1)
        predict = tf.cast(tf.argmax(probability, axis=1), dtype=tf.int32)
        match = tf.cast(tf.equal(predict, y), dtype=tf.int32)
        correct += tf.reduce_sum(match)
        total += y.shape[0]
    return correct / total


def predict(x, args):
    w1, b1, w2, b2, w3, b3 = args
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 1])
    out = network(x, w1, b1, w2, b2, w3, b3)
    probability = tf.nn.softmax(out, axis=1)
    print("predict {}: {}".format(x, probability))


def main():
    x_train, y_train, x_test, y_test = num_dataset.load_dataset()
    # 输入数据范围很重要
    # x_train, y_train, x_test, y_test = x_train/10, y_train, x_test/10, y_test
    args = train(x_train, y_train)
    print("acc: {}%".format(float(test(x_test, y_test, args)) * 100))
    predict(-1000, args)
    predict(-2, args)
    predict(-3, args)


if __name__ == "__main__":
    main()

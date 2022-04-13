import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers, metrics

"""
https://www.youtube.com/watch?v=IJASOYa-jbY&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=72
"""


def build_model():
    model = Sequential([
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(10),
    ])

    model.build(input_shape=[None, 28 * 28])
    model.summary()
    return model


def build_opt():
    return optimizers.Adam(learning_rate=1e-3)


def train_and_test(x_train, y_train, x_test, y_test):
    model = build_model()
    opt = build_opt()
    for epoch in range(30):
        train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
        for step, (x, y) in enumerate(train_db):
            x = tf.reshape(x, [-1, x_train.shape[1] * x_train.shape[2]])
            y = tf.one_hot(y, depth=10)
            grads, loss = train_iter(x, y, model)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            acc = test(x_test, y_test, model)
            if step % 100 == 0:
                print("epoch={}, step={}, loss={}, acc={}".format(epoch, step, loss, acc))


def train_iter(x, y, model):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_mse = tf.reduce_mean(tf.losses.MSE(y, logits))
        loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))
    grads = tape.gradient(loss_ce, model.trainable_variables)
    return grads, loss_ce


def test(x_test, y_test, model):
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
    acc = metrics.Accuracy()
    acc.reset_state()
    for i, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, x_test.shape[1] * x_test.shape[2]])
        out = model(x)
        predict = tf.cast(tf.argmax(tf.nn.softmax(out, axis=1), axis=1), dtype=tf.int32)
        acc.update_state(y, predict)
    return acc.result()


def main():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    train_and_test(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()

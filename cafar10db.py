import tensorflow as tf
from tensorflow.keras import datasets


def data_transform(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255.0 - 1.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def load_db(batch_size=128):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    y_train = tf.one_hot(tf.squeeze(y_train), depth=100)
    y_test = tf.one_hot(tf.squeeze(y_test), depth=100)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.map(data_transform).shuffle(10000).batch(batch_size)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(data_transform).batch(batch_size)
    return train_db, test_db

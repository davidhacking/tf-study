import tensorflow as tf
from tensorflow.keras import datasets


def load_db():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.0
    train_db = tf.data.Dataset.from_tensor_slices(x_train)
    train_db = train_db.shuffle(10000).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    test_db = test_db.batch(128)
    return train_db, test_db

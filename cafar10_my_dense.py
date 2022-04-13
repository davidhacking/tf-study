import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, Sequential, optimizers
import cafar10db

"""
https://www.youtube.com/watch?v=IKtsrptCPhk&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=82
"""


class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight("w", [input_dim, output_dim])
        self.bias = self.add_variable("b", [output_dim])

    def call(self, input_data, *args, **kwargs):
        return input_data @ self.kernel + self.bias


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = MyDense(32 * 32 * 3, 1024)
        self.layer2 = MyDense(1024, 512)
        self.layer3 = MyDense(512, 256)
        self.layer4 = MyDense(256, 128)
        self.out = MyDense(128, 100)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = self.layer1(inputs)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        x = tf.nn.relu(x)
        x = self.layer3(x)
        x = tf.nn.relu(x)
        x = self.layer4(x)
        x = tf.nn.relu(x)
        x = self.out(x)
        return x


def build_model():
    model = MyModel()
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


model_path = 'out_model/my_dense/my_dense.model'


def train(train_db, test_db):
    model = build_model()
    model.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)
    model.evaluate(test_db)
    model.save_weights(model_path)


def test(test_db):
    model = build_model()
    model.load_weights(model_path)
    model.evaluate(test_db)


def main():
    train_db, test_db = cafar10db.load_db()
    train(train_db, test_db)
    # test(test_db)


if __name__ == '__main__':
    main()

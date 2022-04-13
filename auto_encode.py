import fashion_db
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential, layers, optimizers
import images_to_one


def main():
    train_db, test_db = fashion_db.load_db()
    model = build_model()
    opt = optimizers.Adam(learning_rate=1e-3)
    for epoch in range(100):
        for step, x_train in enumerate(train_db):
            x_train = tf.reshape(x_train, [-1, 784])
            with tf.GradientTape() as tape:
                reconstruct = model(x_train)
                loss = tf.losses.binary_crossentropy(x_train, reconstruct, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print("epoch:{}, step:{}, loss: {}".format(epoch, step, loss))

        t_db = iter(test_db)
        x = next(t_db)
        x_hat = tf.sigmoid(model(tf.reshape(x, [-1, 784])))
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        x = x.numpy() * 255.0
        x = x.astype(np.uint8)
        x_hat = x_hat.numpy() * 255.0
        x_hat = x_hat.astype(np.uint8)
        import os
        if not os.path.exists("ae_imgs"):
            os.mkdir("ae_imgs")
        images_to_one.compare_imgs(x, x_hat, "ae_imgs/rec_epoch{}.png".format(epoch))


class AutoEncode(Model):
    def __init__(self, h_dim=20):
        super(AutoEncode, self).__init__()
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim),
        ])
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784),
        ])

    def call(self, inputs, training=None, mask=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat


def build_model():
    model = AutoEncode(h_dim=10)
    model.build(input_shape=(None, 784))
    model.summary()
    return model


if __name__ == '__main__':
    main()

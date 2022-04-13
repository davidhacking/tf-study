import fashion_db
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential, layers, optimizers
import images_to_one

"""
https://www.youtube.com/watch?v=J-JG3Q2Ckyw&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=139
"""

path = "vae_imgs"


def main():
    train_db, test_db = fashion_db.load_db()
    model = build_model()
    opt = optimizers.Adam(learning_rate=1e-3)
    for epoch in range(100):
        for step, x_train in enumerate(train_db):
            x_train = tf.reshape(x_train, [-1, 784])
            with tf.GradientTape() as tape:
                reconstruct, mean, log_var = model(x_train)
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_train, logits=reconstruct)
                rec_loss = tf.reduce_sum(rec_loss) / x_train.shape[0]
                # compute kl divergence (mean, log_var) ~ N(0, 1)
                kl_div = -0.5 * (log_var + 1 - mean ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / x_train.shape[0]
                loss = rec_loss + 10 * kl_div
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print("epoch:{}, step:{}, rec_loss: {}, "
                      "kl_div: {}, loss: {}".format(epoch, step, rec_loss, kl_div, loss))

        t_db = iter(test_db)
        x = next(t_db)
        x_hat, _, _ = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(x_hat)
        x_hat = tf.reshape(x_hat, [-1, 28, 28])
        x = x.numpy() * 255.0
        x = x.astype(np.uint8)
        x_hat = x_hat.numpy() * 255.0
        x_hat = x_hat.astype(np.uint8)
        import os
        if not os.path.exists(path):
            os.mkdir(path)
        images_to_one.compare_imgs(x, x_hat, "{}/rec_epoch{}.png".format(path, epoch))


class VariationalAutoEncode(Model):
    def __init__(self, z_dim=10):
        super(VariationalAutoEncode, self).__init__()
        # encoder
        self.encoder_layer1 = layers.Dense(256)
        self.encoder_layer2 = layers.Dense(128)
        self.mean_layer = layers.Dense(z_dim)
        self.var_layer = layers.Dense(z_dim)

        # decoder
        self.decoder_layer1 = layers.Dense(128)
        self.decoder_layer2 = layers.Dense(256)
        self.out = layers.Dense(784)

    def encode(self, x):
        x = tf.nn.relu(self.encoder_layer1(x))
        x = tf.nn.relu(self.encoder_layer2(x))
        mean = self.mean_layer(x)
        log_var = self.var_layer(x)
        return mean, log_var

    def decode(self, x):
        x = tf.nn.relu(self.decoder_layer1(x))
        x = tf.nn.relu(self.decoder_layer2(x))
        return self.out(x)

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(log_var.shape)
        std_var = tf.exp(log_var) ** 0.5
        return mean + std_var * eps

    def call(self, inputs, training=None, mask=None):
        mean, log_var = self.encode(inputs)
        # reparameterization trick
        z = self.reparameterize(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


def build_model():
    model = VariationalAutoEncode(z_dim=10)
    model.build(input_shape=(4, 784))
    model.summary()
    return model


if __name__ == '__main__':
    main()

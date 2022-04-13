import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

"""
https://www.youtube.com/watch?v=-twrFTKFDsw&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=120
"""

batchsz = 128
total_words = 10000
max_len = 80
embedding_len = 100


def load_db():
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batchsz, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchsz, drop_remainder=True)
    print("db_train={}, db_test={}, \n"
          "x_min={}, x_mean={}, x_max={}, \n"
          "y_min={}, y_mean={}, y_max={}".format(
        (x_train.shape, y_train.shape),
        (x_test.shape, y_test.shape),
        tf.reduce_min(x_train), tf.reduce_mean(x_train), tf.reduce_max(x_train),
        tf.reduce_min(y_train), tf.reduce_mean(y_train), tf.reduce_max(y_train)
    ))
    return db_train, db_test


class MyRNN(keras.Model):
    def __init__(self, h_dim=64, network='rnn', dropout=0.5):
        """
        :param h_dim:
        :param network: 'rnn' 'lstm' 'gru'
        """
        super(MyRNN, self).__init__()
        if network == 'lstm':
            self.state0 = [tf.zeros([batchsz, h_dim]), tf.zeros([batchsz, h_dim])]
            self.state1 = [tf.zeros([batchsz, h_dim]), tf.zeros([batchsz, h_dim])]
        else:
            self.state0 = [tf.zeros([batchsz, h_dim])]
            self.state1 = [tf.zeros([batchsz, h_dim])]

        # [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_len)
        # [b, 80, 100] => [b, 64]
        if network == 'gru':
            self.rnn_cell0 = layers.GRUCell(h_dim, dropout=dropout)
            self.rnn_cell1 = layers.GRUCell(h_dim, dropout=dropout)
        elif network == 'lstm':
            self.rnn_cell0 = layers.LSTMCell(h_dim, dropout=dropout)
            self.rnn_cell1 = layers.LSTMCell(h_dim, dropout=dropout)
        else:
            self.rnn_cell0 = layers.SimpleRNNCell(h_dim, dropout=dropout)
            self.rnn_cell1 = layers.SimpleRNNCell(h_dim, dropout=dropout)
        # [b, 64] => [b]
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        state0 = self.state0
        state1 = self.state1
        out0, out1 = None, None
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        out = self.fc(out1)
        return tf.sigmoid(out)


def main():
    db_train, db_test = load_db()
    model = MyRNN(network='gru')
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )
    model.fit(db_train, epochs=5, validation_data=db_test)
    model.evaluate(db_test)


if __name__ == '__main__':
    main()

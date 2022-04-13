import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np


def main():
    method_name = tf1.saved_model.signature_constants.PREDICT_METHOD_NAME
    rowkeys_file = "vocab_file/rowkeys.vocab"
    embedding_file = "vocab_file/emb.vocab"
    vocab_size = 1000
    num_oov_buckets = 200
    emb = tf.Variable(np.loadtxt(embedding_file, delimiter=' '), dtype=tf.float32)
    rowkeys = tf.tensor
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            rowkeys_file,
            key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter="\n"),
        num_oov_buckets)
    rowkey_ids = table.lookup(rowkeys)


if __name__ == '__main__':
    main()

import datetime
import os

import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.framework import ops
import time
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    di2s = tf.identity(tf.diag_part(kernel_matrix))
    selected_items = []
    selected_item = tf.argmax(di2s)
    selected_items.append(selected_item)
    cis = []
    items_size = tf.shape(di2s)[0]

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        di_optimal = tf.math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        if k > 0:
            ci_optimal = tf.reshape(tf.gather(cis[:k], selected_item, axis=1), [-1, k])
            x = tf.stack(cis[:k])
            eis = tf.squeeze((elements - tf.matmul(ci_optimal, x)) / di_optimal)
        else:
            eis = tf.squeeze(elements / di_optimal)

        cis.append(eis)
        di2s = tf.subtract(di2s, tf.square(eis))
        selected_item = tf.argmax(di2s)

        def true_fn():
            selected_items.append(selected_item)
            return max_length

        def false_fn():
            max_length = len(selected_items)
            return max_length

        tf.cond(di2s[selected_item] < epsilon, true_fn, false_fn)
    truncate_size = tf.cond(items_size < max_length, lambda: items_size, lambda: max_length)
    res = tf.squeeze(tf.slice([tf.stack(selected_items)], [0, 0], [-1, truncate_size]))
    return res


def signature(function_dict):
    signature_dict = {}
    for k, v in function_dict.items():
        inputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in v['inputs'].items()}
        outputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in v['outputs'].items()}
        signature_dict[k] = tf.saved_model.build_signature_def(inputs=inputs, outputs=outputs,
                                                               method_name=v['method_name'])
    return signature_dict


def export():
    cur = os.path.join("out_model", str(int(time.time())))
    method_name = "prediction"
    vocab_dir = "vocab_file"
    rowkeys_file = os.path.join(vocab_dir, "rowkeys.vocab")
    embedding_file = os.path.join(vocab_dir, "emb.vocab")
    max_length = 100
    num_oov_buckets = 200
    vocab_size = 1000

    with tf.Graph().as_default(), tf.Session() as sess:
        # placeholder 表示参数名
        rowkeys = tf.placeholder(dtype=tf.key, name="rowkeys")
        algorithm_ids = tf.placeholder(dtype=tf.uint32, name="algorithm_id")
        scores = tf.placeholder(dtype=tf.float32, name="scores")
        emb_value = np.loadtxt(embedding_file, delimiter=' ')
        emb = tf.Variable(emb_value, dtype=tf.float32)
        table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                rowkeys_file,
                key_dtype=tf.key, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                delimiter="\n"),
            vocab_size - num_oov_buckets)

        rowkeys = tf.constant(["999621e2b53751aw",
                               "999621e44d8817bk",
                               "999621efd86526aw",
                               "999622231d5654ah",
                               "999621f5591306bw"], dtype=tf.key)
        algorithm_ids = tf.constant([2081,
                                     2803,
                                     2086,
                                     2803,
                                     2086], dtype=tf.uint32)
        scores = tf.constant([0.1,
                              0.2,
                              0.3,
                              0.11,
                              0.7
                              ], dtype=float)

        rowkey_ids = table.lookup(rowkeys)
        rowkeys_embedding = tf.nn.embedding_lookup(emb, rowkey_ids)
        rk_emb_norm = tf.linalg.norm(rowkeys_embedding, axis=1, keepdims=True)
        rowkeys_embedding /= rk_emb_norm
        # 将nan值替换成0值 tf.where(condition, x, y) condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
        # rowkeys_embedding shape=(5, 200)
        rowkeys_embedding = tf.where(tf.is_nan(rowkeys_embedding), tf.zeros_like(rowkeys_embedding), rowkeys_embedding)
        similarities = tf.cast(tf.matmul(rowkeys_embedding, rowkeys_embedding, transpose_b=True), tf.float32)
        # tf.reshape(scores, [-1, 1]) 这样不是nx1的矩阵？
        related_x = tf.reshape(scores, [-1, 1])
        related_x_t = tf.reshape(scores, [1, -1])
        kernel_matrix = related_x * similarities * related_x_t
        indices = dpp(kernel_matrix, max_length)
        predict_rowkeys = tf.gather(rowkeys, indices)
        predict_scores = tf.gather(scores, indices)
        predict_algorithm_ids = tf.gather(algorithm_ids, indices)
        predict_positions = indices

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        signature_def_map = signature({
            "prediction": {
                "inputs": {
                    'rowkeys': rowkeys,
                    "scores": scores,
                    "algorithm_ids": algorithm_ids
                },
                "outputs": {
                    "rowkeys": predict_rowkeys,
                    "scores": predict_scores,
                    "algorithm_ids": predict_algorithm_ids,
                    "origin_position": predict_positions
                },
                "method_name": method_name
            },
        })

        builder = tf.saved_model.builder.SavedModelBuilder(cur)
        builder.add_meta_graph_and_variables(sess, tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map,
                                             assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
                                             main_op=tf.tables_initializer())
        builder.save()

        os.mkdir(os.path.join(cur, "assets.extra"))
        with tf.python_io.TFRecordWriter(os.path.join(cur, "assets.extra/tf_serving_warmup_requests")) as writer:
            request = predict_pb2.PredictRequest(
                model_spec=model_pb2.ModelSpec(name="union_dpp_model_serving_kbsv", signature_name="prediction"),
                inputs={
                    "rowkeys": tf.make_tensor_proto(["999622118c2134ah", "9996220dfc0929bk"], dtype=tf.key),
                    "algorithm_ids": tf.make_tensor_proto([2081, 2081], dtype=tf.uint32),
                    "scores": tf.make_tensor_proto([0.7, 0.7], dtype=tf.float32)
                }
            )
            print(request)
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request))
            writer.write(log.SerializeToString())


if __name__ == '__main__':
    export()

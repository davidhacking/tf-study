import datetime
import tensorflow as tf

"""
tensorboard --logdir logs
"""


def build_tensorboard():
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/" + cur_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer

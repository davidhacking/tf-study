import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    print('GPU', tf.config.list_physical_devices('GPU'))
    a = tf.convert_to_tensor([[[1, 2], [0, 3]], [[4, 5], [6, 6]]])
    print(a.shape)
    b = tf.reduce_max(a, axis=1)
    print(b.shape, b)

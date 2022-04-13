import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, Sequential, optimizers
import cafar10db

"""
https://www.youtube.com/watch?v=cWcyfIX65lA&list=PLh7DRwYmUgh7swOvZUZ52LMeGDmjFH0nv&index=110
"""


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        if strides > 1:
            self.downsample = Sequential([])
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=strides))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, *args, **kwargs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        out = layers.add([out, identity])
        out = tf.nn.relu(out)
        return out


class ResNet(keras.Model):
    def __init__(self, layer_dims, class_num=100):
        super(ResNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same")
        ])
        self.layer1 = self.build_blocks(64, layer_dims[0])
        self.layer2 = self.build_blocks(128, layer_dims[1], strides=2)
        self.layer3 = self.build_blocks(256, layer_dims[2], strides=2)
        self.layer4 = self.build_blocks(512, layer_dims[3], strides=2)

        # 用于转换
        self.avg_pool = layers.GlobalAveragePooling2D()
        # 最后的全连接层
        self.fc = layers.Dense(class_num)

    def call(self, inputs):
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avg_pool(outputs)
        outputs = self.fc(outputs)
        return outputs

    def build_blocks(self, filter_num, block_num, strides=1):
        blocks = Sequential([])
        blocks.add(BasicBlock(filter_num, strides))

        for _ in range(1, block_num):
            blocks.add(BasicBlock(filter_num, strides=1))
        return blocks


def build_resnet18():
    return ResNet([2, 2, 2, 2])


def main():
    train_db, test_db = cafar10db.load_db()
    model = build_resnet18()
    model.build(input_shape=[None, 32, 32, 3])
    model.summary()
    opt = optimizers.Adam(learning_rate=1e-4)
    for epoch in range(50):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model.call(x)
                loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                acc = test(test_db, model)
                print("epoch={}, step={}, loss={}, acc={}".format(epoch, step, float(loss), acc))


def test(test_db, model):
    total, correct = 0, 0
    for x, y in test_db:
        logits = model(x)
        pred = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        y = tf.cast(tf.argmax(y, axis=1), dtype=tf.int32)
        equals = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
        correct += int(equals)
        total += x.shape[0]
    return correct / total


if __name__ == '__main__':
    main()

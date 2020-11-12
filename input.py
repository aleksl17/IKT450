# TODO
# loop through files
# Read/write files
# remake file

import tensorflow as tf

def input(amount):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

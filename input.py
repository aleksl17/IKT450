# TODO
# loop through files
# Read/write files
# remake file

import numpy
import tensorflow as tf
from image_shuffler import image_shuffler


def img_input(amount):
    x_train = []
    x_test = []
    y_train_shuffle = []
    y_test_shuffle = []
    (x_train_input, y_train_classify_input), (x_test_input, y_test_classify_input) = tf.keras.datasets.cifar10.load_data()
    for a in range(amount):
        x_train_image_shuffler, y_train_image_shuffler = image_shuffler(x_train_input[a])
        x_train.append(x_train_image_shuffler)
        y_train_shuffle.append(y_train_image_shuffler)
    for b in range(amount // 5):
        x_test_image_shuffler, y_test_shuffle_image_shuffler = image_shuffler(x_test_input[b])
        x_train.append(x_test_image_shuffler)
        y_train_shuffle.append(y_test_shuffle_image_shuffler)
    return x_train, y_train_classify_input, y_train_shuffle, x_test, y_test_classify_input, y_test_shuffle


x_train, y_train_classify, y_train_shuffle, x_test, y_test_classify, y_test_shuffle = img_input(2)

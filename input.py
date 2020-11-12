# TODO
# loop through files
# Read/write files
# remake file

import numpy
import tensorflow as tf
from image_shuffler import image_shuffler


def img_input(amount):
    (x_train, y_train_classify), (x_test, y_test_classify) = tf.keras.datasets.cifar10.load_data()
    for a in range(amount):
        x_train, y_train_shuffle = image_shuffler(x_train[a])
    for b in range((amount // 5)):
        x_test, y_test_shuffle = image_shuffler(x_test[b])
    return x_train, y_train_classify, y_train_shuffle, x_test, y_test_classify, y_test_shuffle


x_train, y_train_classify, y_train_shuffle, x_test, y_test_classify, y_test_shuffle = img_input(9)
# print("x shape: ", x_train.shape)
# print("x: ", x_train[2])
# print("y shuffle: ", y_train_shuffle)

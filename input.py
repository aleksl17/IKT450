# TODO
# loop through files
# Read/write files
# remake file

import numpy
import tensorflow as tf
from matplotlib import pyplot
from image_shuffler import image_shuffler
from image_shuffler import crop_centered


def img_input(amount):
    y_train_shuffled = []
    y_test_shuffled = []
    x_train_shuffled = []
    x_test_shuffled = []
    x_train_cropped = []
    x_test_cropped = []

    (x_train_input, y_train_classify_input), (x_test_input, y_test_classify_input) = tf.keras.datasets.cifar10.load_data()

    for a in range(amount):
        x_train_image_shuffler, y_train_image_shuffler = image_shuffler(x_train_input[a])
        x_train_shuffled.append(x_train_image_shuffler)
        y_train_shuffled.append(y_train_image_shuffler)
        x_train_cropped.append(crop_centered(x_train_input[a]))

    for b in range(amount // 5):
        x_test_image_shuffler, y_test_shuffle_image_shuffler = image_shuffler(x_test_input[b])
        x_test_shuffled.append(x_test_image_shuffler)
        y_test_shuffled.append(y_test_shuffle_image_shuffler)
        x_test_cropped.append(crop_centered(x_test_input[b]))

    return x_train_cropped, x_train_shuffled, y_train_classify_input, y_train_shuffled, x_test_cropped, x_test_shuffled, y_test_classify_input, y_test_shuffled


x_train, x_train_shuffle, y_train_classify, y_train_shuffle, x_test, x_test_shuffle, y_test_classify, y_test_shuffle = img_input(9)

pyplot.subplot(211)
pyplot.imshow(x_train[1])
pyplot.subplot(212)
pyplot.imshow(x_train_shuffle[1])
pyplot.show()

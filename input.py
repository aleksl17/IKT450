# TODO
# Make processed dataset file and file management

import tensorflow as tf
from matplotlib import pyplot
from image_shuffler import image_shuffler, crop_centered


# Imports CIFAR10 dataset. Then performes cropping and shuffeling of dataset.
def img_input(amount):
    test_amount = 1
    if amount <= -1:
        amount = 50000
        test_amount = 10000
    elif amount <= 5:
        test_amount = 1
    elif amount > 50000:
        amount = 50000
        test_amount = 10000
    else:
        test_amount = amount // 5

    y_train_shuffled, y_test_shuffled, x_train_shuffled, x_test_shuffled, x_train_cropped, x_test_cropped = ([] for l in range(6))

    (x_train_input, y_train_classify_input), (x_test_input, y_test_classify_input) = tf.keras.datasets.cifar10.load_data()

    for atr in range(amount):
        x_train_image_shuffler, y_train_image_shuffler = image_shuffler(x_train_input[atr])
        x_train_shuffled.append(x_train_image_shuffler)
        y_train_shuffled.append(y_train_image_shuffler)
        x_train_cropped.append(crop_centered(x_train_input[atr]))

    for ate in range(test_amount):
        x_test_image_shuffler, y_test_image_shuffler = image_shuffler(x_test_input[ate])
        x_test_shuffled.append(x_test_image_shuffler)
        y_test_shuffled.append(y_test_image_shuffler)
        x_test_cropped.append(crop_centered(x_test_input[ate]))

    return x_train_cropped, x_train_shuffled, y_train_classify_input, y_train_shuffled, x_test_cropped, x_test_shuffled, y_test_classify_input, y_test_shuffled


# Syntax example and dataset visualization example.
x_train, x_train_shuffle, y_train_classify, y_train_shuffle, x_test, x_test_shuffle, y_test_classify, y_test_shuffle = img_input(25)

pyplot.subplot(211)
pyplot.imshow(x_train[1])
pyplot.subplot(212)
pyplot.imshow(x_train_shuffle[1])
pyplot.show()

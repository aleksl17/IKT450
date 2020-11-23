import numpy
import tensorflow as tf
from image_shuffler import image_shuffler, crop_centered, numbers_to_binary


# Imports CIFAR10 dataset. Then performs cropping and shuffling of dataset.
def img_input(amount):
    length = 10
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

    y_train_shuffled, y_test_shuffled, y_train_classify_output, y_test_classify_output, x_train_shuffled, x_test_shuffled, x_train_cropped, x_test_cropped = ([] for l in range(8))

    (x_train_input, y_train_classify_input), (x_test_input, y_test_classify_input) = tf.keras.datasets.cifar10.load_data()

    for atr in range(amount):
        x_train_image_shuffler, y_train_image_shuffler = image_shuffler(x_train_input[atr])
        x_train_shuffled.append(x_train_image_shuffler)
        y_train_shuffled.append(y_train_image_shuffler)
        x_train_cropped.append(crop_centered(x_train_input[atr]))
        y_train_classify_output.append(numbers_to_binary(y_train_classify_input[atr], length))

    for ate in range(test_amount):
        x_test_image_shuffler, y_test_image_shuffler = image_shuffler(x_test_input[ate])
        x_test_shuffled.append(x_test_image_shuffler)
        y_test_shuffled.append(y_test_image_shuffler)
        x_test_cropped.append(crop_centered(x_test_input[ate]))
        y_test_classify_output.append(numbers_to_binary(y_test_classify_input[ate], length))

    # Convert from list to numpy array.
    x_train_cropped = numpy.array(x_train_cropped)
    x_train_shuffled = numpy.array(x_train_shuffled)
    y_train_classify_output = numpy.array(y_train_classify_output)
    y_train_shuffled = numpy.array(y_train_shuffled)
    x_test_cropped = numpy.array(x_test_cropped)
    x_test_shuffled = numpy.array(x_test_shuffled)
    y_test_classify_output = numpy.array(y_test_classify_output)
    y_test_shuffled = numpy.array(y_test_shuffled)

    return x_train_cropped, x_train_shuffled, y_train_classify_output, y_train_shuffled, x_test_cropped, x_test_shuffled, y_test_classify_output, y_test_shuffled

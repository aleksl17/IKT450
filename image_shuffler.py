import numpy
import copy
import random
import tensorflow as tf
from PIL import Image
from itertools import product
from matplotlib import pyplot

# Configurable variables
crop_amount = 9  # Set to "-1" to crop ALL images.
random.seed(69)

# Variables
ds = []


# Crops 3-dimensional image arrays towards center.
def crop_centered(img, crop_x, crop_y):
    x, y, c = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_x:start_x + crop_x, start_y:start_y + crop_y, :]


# Shuffles image in a 3x3 fashion.
def shuffle_image(img):
    sections = []
    # Divide picture into
    print("Shape: ", img.shape)
    for px in range(3):
        for py in range(3):
            section = img[10*px:10*px+10, 10*py:10*py+10]
            sections.append(section)
    random.shuffle(sections)
    shuffle = numpy.array(sections)
    print("Shape: ", shuffle.shape)
    return shuffle


    # section1 = []
    # section2 = []
    # section3 = []
    # print("Sections: ", sections)
    # print("Shape: ", img.shape)
    # print("Length: ", len(img))
    # print("Before: ", img)

    # pyplot.subplot(211)
    # pyplot.imshow(img)
    # pyplot.subplot(212)
    # pyplot.imshow(sections[8])
    # pyplot.show()
    # shuffle = numpy.array(sections)

    # print("Shape: ", shuffle.shape)
    # print("Length: ", len(shuffle))
    # print("After: ", shuffle)

    # x_len = range(0, x, img_pieces)
    # y_len = range(0, y, img_pieces)
    # print(x_len)
    # sections = ()
    #
    # sections = ((i, j, i + img_pieces, j + img_pieces) for i, j, in product(x_len, y_len))
    # section = img[10 * p:10 * p + 10, 10 * p:10 * p + 10][10 * p:10 * p + 10, 10 * p:10 * p + 10]
    # print(sections)
    #
    # pieces = [img.crop(section) for section in sections]
    # random.shuffle(pieces)
    # return sections, len(x_len), len(y_len)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Crop Images
if crop_amount == -1:
    crop_amount = len(x_train)
for i in range(crop_amount):
    tmp_crop_img = crop_centered(x_train[i], 30, 30)
    ds.append(tmp_crop_img)

bs = shuffle_image(ds[1])

pyplot.subplot(211)
pyplot.imshow(ds[1])
pyplot.subplot(212)
pyplot.imshow(bs[1])
pyplot.show()

# Shuffle images
# tmp_img = ds[1].copy
# sds = shuffle_image(tmp_img, 3)
# pyplot.imshow(sds)
# pyplot.show()


# pyplot.subplot(211)
# pyplot.imshow(x_train[1])
# pyplot.subplot(212)
# pyplot.imshow(ds[1])
# pyplot.show()

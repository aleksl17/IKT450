import numpy
import copy
import random
import tensorflow as tf
from PIL import Image
from itertools import product
from matplotlib import pyplot

# Configurable variables
crop_amount = 9  # Set to "-1" to crop ALL images.

# Variables
ds = []


# Crops 3-dimensional image arrays towards center.
def crop_centered(img, crop_x, crop_y):
    x, y, c = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_x:start_x + crop_x, start_y:start_y + crop_y, :]


def shuffle_image(img):
    sections = []
    print("Sections: ", sections)
    print(img.shape)
    print("Before: ", img)
    for p in range(9):
        section = img[10 * p:10 * p + 10, 10 * p:10 * p + 10, :][10 * p:10 * p + 10, 10 * p:10 * p + 10, :]
        sections.extend(section)
    shuffle = numpy.array(sections)
    print(shuffle.shape)
    print("After: ", shuffle)
    return shuffle

    # x_len = range(0, x, img_pieces)
    # y_len = range(0, y, img_pieces)
    # print(x_len)
    # sections = ()
    #
    # sections = ((i, j, i + img_pieces, j + img_pieces) for i, j, in product(x_len, y_len))
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


# shuffle_image(ds[1])

pyplot.subplot(211)
pyplot.imshow(ds[1])
pyplot.subplot(212)
pyplot.imshow(shuffle_image(ds[1]))
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

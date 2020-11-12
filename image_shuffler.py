import numpy
import random
import tensorflow as tf

# Configurable Global Variables
crop_amount = 9  # Set to "-1" to crop ALL images.
random.seed(69)

# Global Variables
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
    full_y = []
    random_head = list(range(0, 10))
    # Divide picture into
    print("Shape: ", img.shape)
    for px in range(3):
        for py in range(3):
            section = img[10*px:10*px+10, 10*py:10*py+10]
            sections.append(section)
    shuffle = numpy.array(sections)
    combine = list(zip(shuffle, random_head))
    random.shuffle(combine)
    shuffle, random_head = zip(*combine)
    for rh in random_head:
        tmp_list = [0]*9
        tmp_list[rh] = 1
        full_y.extend(tmp_list)
    return shuffle, full_y


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Crop Images
if crop_amount == -1:
    crop_amount = len(x_train)
for i in range(crop_amount):
    tmp_crop_img = crop_centered(x_train[i], 30, 30)
    ds.append(tmp_crop_img)

bs, y = shuffle_image(ds[1])

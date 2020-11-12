import numpy
import random
import tensorflow as tf

# Configurable Global Variables
crop_amount = 9  # Set to "-1" to crop ALL images. # MOVE TO INPUT
random.seed(69)

# Global Variables
ds = []


# Crops 3-dimensional image arrays towards center.
def crop_centered(img):
    x, y, c = img.shape
    start_x = x//2-(30//2)
    start_y = y//2-(30//2)
    return img[start_x:start_x+30, start_y:start_y+30, :]


# Shuffles image in a 3x3 fashion.
def shuffle_image(img):
    sections = []
    full_y = []
    random_head = list(range(0, 10))
    # Divide 30x30 picture into 9, 10x10 sub-pictures
    for px in range(3):
        for py in range(3):
            section = img[10*px:10*px+10, 10*py:10*py+10]
            sections.append(section)
    # Shuffle images
    shuffle = numpy.array(sections)
    combine = list(zip(shuffle, random_head))
    random.shuffle(combine)
    shuffle, random_head = zip(*combine)
    # Convert number y to binary y.
    for rh in random_head:
        tmp_list = [0]*9
        tmp_list[rh] = 1
        full_y.extend(tmp_list)
    return shuffle, full_y


def image_shuffler(img):
    cropped_img = crop_centered(img)
    x_train_shuffled, y_train_shuffled = shuffle_image(cropped_img)
    return x_train_shuffled, y_train_shuffled

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Crop Images
if crop_amount == -1:
    crop_amount = len(x_train)
for i in range(crop_amount):
    tmp_crop_img = crop_centered(x_train[i])
    ds.append(tmp_crop_img)

bs, smol_y = shuffle_image(ds[1])

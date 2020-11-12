import numpy
import random
import tensorflow as tf

# Configurable Global Variables
# random.seed(69)


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
    # Convert array of numbers to binary array
    for rh in random_head:
        tmp_list = [0]*9
        tmp_list[rh] = 1
        full_y.extend(tmp_list)
    return shuffle, full_y


def image_shuffler(img):
    cropped_img = crop_centered(img)
    x_train_shuffled, y_train_shuffled = shuffle_image(cropped_img)
    return x_train_shuffled, y_train_shuffled

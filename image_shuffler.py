import numpy
import random


# Crops 3-dimensional image arrays towards a 30x30 center.
def crop_centered(img):
    x, y, c = img.shape
    start_x = x // 2 - (30 // 2)
    start_y = y // 2 - (30 // 2)
    return img[start_x:start_x + 30, start_y:start_y + 30, :]


# Shuffles image in a 3x3 fashion.
def shuffle_image(img):
    full_y = []
    random_head = list(range(0, 10))

    # Deconstruct image via function
    sections = deconstruct_image(img)

    # Shuffle images with labels
    combine = list(zip(sections, random_head))
    random.shuffle(combine)
    sections, random_head = zip(*combine)

    # Convert array of numbers to binary location-based array.
    for rh in random_head:
        tmp_list = [0] * 9
        tmp_list[rh] = 1
        full_y.extend(tmp_list)

    # Reconstruct image via function.
    reconstruct = reconstruct_image(sections)

    return reconstruct, full_y


# Deconstruct a single array item into a 9-long 10, 10, 3 image array.
def deconstruct_image(img_item):
    sections = []

    # Divide 30x30 picture into 9, 10x10 sub-pictures.
    for px in range(3):
        for py in range(3):
            section = img_item[10 * px:10 * px + 10, 10 * py:10 * py + 10]
            sections.append(section)

    # Convert list to numpy array
    sections = numpy.array(sections)

    return sections


# Reconstructs a 9-long list of 10, 10, 3 images into a single 30, 30, 3 array item.
def reconstruct_image(img_array):
    first_two = numpy.concatenate((img_array[0], img_array[1]), axis=1)
    first_row = numpy.concatenate((first_two, img_array[2]), axis=1)
    second_two = numpy.concatenate((img_array[3], img_array[4]), axis=1)
    second_row = numpy.concatenate((second_two, img_array[5]), axis=1)
    third_two = numpy.concatenate((img_array[6], img_array[7]), axis=1)
    third_row = numpy.concatenate((third_two, img_array[8]), axis=1)
    first_two_row = numpy.concatenate((first_row, second_row), axis=0)
    reconstruct = numpy.concatenate((first_two_row, third_row), axis=0)

    # Convert reconstructed image list to numpy array.
    reconstruct = numpy.array(reconstruct)

    return reconstruct


# Crops and shuffles input image.
def image_shuffler(img):
    cropped_img = crop_centered(img)
    x_train_shuffled, y_train_shuffled = shuffle_image(cropped_img)
    return x_train_shuffled, y_train_shuffled

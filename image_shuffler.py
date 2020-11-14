import numpy
import random

# TODO
# Move "deconstruct" and "reconstruct" code into functions
# Optimize "de/reconstruct" code with loop
# Consider moving labeling into separate function
# Verify label code


# Crops 3-dimensional image arrays towards a 30x30 center.
def crop_centered(img):
    x, y, c = img.shape
    start_x = x // 2 - (30 // 2)
    start_y = y // 2 - (30 // 2)
    return img[start_x:start_x + 30, start_y:start_y + 30, :]


# Shuffles image in a 3x3 fashion.
def shuffle_image(img):
    sections = []
    full_y = []
    random_head = list(range(0, 10))

    # Divide 30x30 picture into 9, 10x10 sub-pictures
    for px in range(3):
        for py in range(3):
            section = img[10 * px:10 * px + 10, 10 * py:10 * py + 10]
            sections.append(section)

    # Convert python list to numpy array.
    all_sections = numpy.array(sections)

    # Shuffle images
    combine = list(zip(all_sections, random_head))
    random.shuffle(combine)
    all_sections, random_head = zip(*combine)

    # Convert array of numbers to binary location-based array
    for rh in random_head:
        tmp_list = [0] * 9
        tmp_list[rh] = 1
        full_y.extend(tmp_list)

    # Reconstruct image
    first_two = numpy.concatenate((all_sections[0], all_sections[1]), axis=1)
    first_row = numpy.concatenate((first_two, all_sections[2]), axis=1)
    second_two = numpy.concatenate((all_sections[3], all_sections[4]), axis=1)
    second_row = numpy.concatenate((second_two, all_sections[5]), axis=1)
    third_two = numpy.concatenate((all_sections[6], all_sections[7]), axis=1)
    third_row = numpy.concatenate((third_two, all_sections[8]), axis=1)
    first_two_row = numpy.concatenate((first_row, second_row), axis=0)
    reconstruct = numpy.concatenate((first_two_row, third_row), axis=0)

    # Convert reconstructed image list to numpy array.
    reconstruct = numpy.array(reconstruct)

    return reconstruct, full_y


# Crops and shuffles input image.
def image_shuffler(img):
    cropped_img = crop_centered(img)
    x_train_shuffled, y_train_shuffled = shuffle_image(cropped_img)
    return x_train_shuffled, y_train_shuffled

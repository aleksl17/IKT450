import numpy
import random


# Crops 3-dimensional image arrays towards a 30x30 center.
def crop_centered(img):
    x, y, c = img.shape
    start_x = x // 2 - (30 // 2)
    start_y = y // 2 - (30 // 2)
    return img[start_x:start_x + 30, start_y:start_y + 30, :]


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


# Crops and shuffles input image.
def image_shuffler(img):
    cropped_img = crop_centered(img)
    x_train_shuffled, y_train_shuffled = shuffle_image(cropped_img)
    return x_train_shuffled, y_train_shuffled


# Reconstructs image from image array and y array.
def image_reconstructor(org_img, reconstruct_y):
    reconstruct_img = []
    for oi in range(len(org_img)):
        # Split image item into 9-long image array.
        deconstructed_reconstruct_img = deconstruct_image(org_img[oi])

        # Combine 9-long image array and y into a single array.
        dri_ry_combined = list(zip(reconstruct_y, deconstructed_reconstruct_img))

        # Order array from low to high based on y.
        ordered_img = [[]]
        i = 0
        while len(dri_ry_combined) > 0:
            if ordered_img[i][i] < lowest:
                lowest = dri_ry_combined[i]
            i += 1
            if i == len(dri_ry_combined):
                ordered_img.append(lowest)
                dri_ry_combined.remove(lowest)
                if dri_ry_combined:
                    lowest = dri_ry_combined[0]
                i = 0

        # Separate y and 9-long image array.
        y, img = zip(*ordered_img)

        reconstruct_ordered_img = reconstruct_image(img)

        reconstruct_img.append(reconstruct_ordered_img)

    # Convert python list into numpy array.
    reconstruct_img = numpy.array(reconstruct_img)

    return reconstruct_img

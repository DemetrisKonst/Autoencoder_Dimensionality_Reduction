import os
import struct
import numpy as np

def filepath_is_not_valid(filepath):
    """ function used to check whether a filepath containing information is valid """

    # check if the path leads to a file
    if not os.path.isfile(filepath):
        # return false
        return True

    # return false since the path is valid
    return False


def filepath_can_be_reached(filepath):
    """ Function used to check if a filepath can be used to create a file """

    # try to open the filepath to write, and if it throws a FileNotFoundError, then it is invalid
    try:
        file = open(filepath, "w")
        file.close()
        # delete it
        os.remove(filepath)
    # exception was thrown, therefore filepath is invalid
    except FileNotFoundError:
        return False

    # return True as the above operations were successful
    return True


def parse_dataset(filepath):
    """ function used to parse the data of a dataset """

    # open the dataset
    with open(filepath, "rb") as dataset:
        # read the magic number and the number of images
        magic_number, number_of_images = struct.unpack(">II", dataset.read(8))
        # read the number of rows and number of columns per image
        rows, columns = struct.unpack(">II", dataset.read(8))
        # now read the rest of the file using numpy.fromfile()
        images = np.fromfile(dataset, dtype=np.dtype(np.uint8).newbyteorder(">"))
        # reshape so that the final shape is (number_of_images, rows, columns)
        images = images.reshape((number_of_images, rows, columns))

    # return the images
    return images


def parse_labelset(filepath):
    """ function used to parse the data of a labelset """

    # open the file
    with open(filepath, "rb") as labelset:
        # read the magic number and the number of labels
        magic_number, number_of_labels = struct.unpack(">II", labelset.read(8))
        # now read the rest of the file using numpy.fromfile()
        labels = np.fromfile(labelset, dtype=np.dtype(np.uint8).newbyteorder(">"))

    # return the labels
    return labels

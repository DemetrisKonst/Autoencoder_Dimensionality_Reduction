import os
import struct
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import save_model, load_model


def parse_input(arg=None, autoencoder=True):
    """ function used to parse the command line input of the autoencoder """

    # create the argument parser
    description = "Python script that creates an autoencoder used to reduce the dimensionality " \
                  "of the MNIST dataset."
    parser = argparse.ArgumentParser(description=description)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the training examples."
    parser.add_argument("-d", "--data", type=str, action="store", metavar="training_examples_path",
                        required=True, help=help)

    # if this is the input for the autoencoder, then we are ok; else parse the rest arguments
    if not autoencoder:

        # add an argument for the path of the training labels
        help = "The full/relative path to the file containing the training labels."
        parser.add_argument("-dl", "--datalabels", type=str, action="store",
                            metavar="training_labels_path", required=True, help=help)

        # add an argument for the path of the testing examples
        help = "The full/relative path to the file containing the testing examples."
        parser.add_argument("-t", "--test", type=str, action="store",
                            metavar="testing_examples_path", required=True, help=help)

        # add an argument for the path of the testing labels
        help = "The full/relative path to the file containing the testing labels."
        parser.add_argument("-tl", "--testlabels", type=str, action="store",
                            metavar="testing_labels_path", required=True, help=help)

        # add an argumrnt for the path of the saved encoder model
        help = "The full/relative path to the .h5 file containing the pre-trained encoder."
        parser.add_argument("-model", "--modelpath", type=str, action="store",
                            metavar="pretrained_encoder_path", required=True, help=help)

    # parse the arguments and return the result
    return parser.parse_args(arg)


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


def save_keras_model(model, model_path):
    """ Function used to save a model in a specific path """

    # just save the model
    save_model(model, model_path)


def load_keras_model(model_path):
    """ Function used to load a model from a specific path """

    # just save the model
    return load_model(model_path)


def print_image(image, rows, columns):
    """ function used to print an image to the console """

    # for each row of the image
    for i in range(rows):
        # for each column
        for j in range(columns):
            # print the value at the coordinate (i, j) if it is not 0
            if image[i, j] != 0:
                print("{:.3f}".format(image[i, j]), end="")
            else:
                print("   ", end="")
        # print a newline since the row has finished
        print()
    print()


def plot_image(image):
    """ fuction used to plot an image using matplotlib """

    # plot and show the image
    plt.imshow(image, cmap="gray")
    plt.show()


def plot_example_images(title, images, label, pred):
    """ fuction used to plot, in one figure, 12 images with their predicted and actual labels """
    fig=plt.figure(figsize=(9, 12))
    fig.suptitle(title, fontsize=14)
    columns = 4
    rows = 3
    for i in range(len(pred)):
        im = fig.add_subplot(rows, columns, i+1)
        title = "Predicted {}, Class {}".format(pred[i][0], label[i][0])
        im.title.set_text(title)
        plt.imshow(images[i], cmap='gray')

    fig.tight_layout()
    plt.show()

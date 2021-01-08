import argparse
import sys
sys.path.append("../../utils")

from utils import filepath_can_be_reached
from error_utils import *
from interface_utils import *

DEFAULT_UNITS = 64

def parse_input(arg=None):
    """ function used to parse the command line input of the reducer """

    # create the argument parser
    description = "Python script that reduces the dimensionality of the MNIST dataset" \
                  "through a previously trained encoder."
    parser = argparse.ArgumentParser(description=description)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the dataset."
    parser.add_argument("-d", "--data", type=str, action="store", metavar="dataset_path",
                        required=True, help=help)

    # add an argument for the path of the output dataset
    help = "The full/relative path to the file to write the produced dataset."
    parser.add_argument("-dl", "--datalabels", type=str, action="store",
                        metavar="datalabels_path", required=True, help=help)

    # add an argument for the path of the output queryset
    help = "The full/relative path to the file to write the produced queryset."
    parser.add_argument("-ol", "--output_path", type=str, action="store",
                        metavar="output_databels_path", required=True, help=help)

    help = "The full/relative path to the .h5 file containing the pre-trained encoder."
    parser.add_argument("-model", "--model_path", type=str, action="store",
                        metavar="pretrained_encoder_path", required=True, help=help)

    # parse the arguments and return the result
    return parser.parse_args(arg)

def get_units():
    """ function used to read the number of units in the fully-connected layer of the classifier """

    # get the number of units
    prompt = "\nGive the number of units in the fully-connected layer (default = {}): "
    prompt = prompt.format(DEFAULT_UNITS)
    units = input(prompt)

    # make sure the user gives a legit input
    while units != "":

        # try to convert the input to an int
        try:
            units = int(units)
            # it must be a positive integer
            if units <= 0:
                raise ValueError
            # if we get here then the input is fine, so break
            break

        # catch error and try again
        except ValueError:
            print("The number of units must a positive integer. Please try again.")
            units = input(prompt)

    # check if the user wants to use the deault value
    if units == "":
        units = DEFAULT_UNITS

    # return the final value
    return units


def get_classification_input():
    """ function used to get the input of the classifier """

    # get the input values one by one
    units = get_units()
    epochs = get_epochs()
    batch_size = get_batch_size()

    # print some newlines and retun the values as a tuple of 3 elements
    print("\n")
    return units, epochs, batch_size

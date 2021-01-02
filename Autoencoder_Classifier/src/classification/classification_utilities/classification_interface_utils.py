import sys
sys.path.append("../../utils")

from utils import filepath_can_be_reached
from error_utils import *
from interface_utils import *

DEFAULT_UNITS = 64

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

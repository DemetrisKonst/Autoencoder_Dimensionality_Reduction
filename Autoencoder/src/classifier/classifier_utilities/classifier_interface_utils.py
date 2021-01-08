import argparse
import sys
sys.path.append("../../utils")

from utils import filepath_can_be_reached
from error_utils import *
from interface_utils import *

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

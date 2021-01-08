import tensorflow as tf
import tensorflow.keras as keras

import logging
import sys

sys.path.append("../utils")
sys.path.append("reduce_utilities")

from utils import *
from interface_utils import *
from reduce_utils import *
from reduce_interface_utils import *

def main(args):
    """ main() driver function """

    # first make sure that the paths to the provided dataset & queryset are valid
    if filepath_is_not_valid(args.dataset):
        logging.error("The path {} is not a file. Aborting..".format(args.dataset))
        exit()

    if filepath_is_not_valid(args.queryset):
        logging.error("The path {} is not a file. Aborting..".format(args.queryset))
        exit()

    # then make sure that the paths to the output files are accessible
    if not filepath_can_be_reached(args.output_dataset):
        logging.error("The path {} cannot be reached to create file. Aborting..".format(args.output_dataset))
        exit()

    if not filepath_can_be_reached(args.output_queryset):
        logging.error("The path {} cannot be reached to create file. Aborting..".format(args.output_queryset))
        exit()

    # get the data from the training set
    dataset = parse_dataset(args.dataset)
    queryset = parse_dataset(args.queryset)
    rows = dataset.shape[1]
    columns = dataset.shape[2]

    # apply preprocessing
    dataset = preprocess(dataset, rows, columns)
    queryset = preprocess(queryset, rows, columns)

    encoder_path = './encoders/z10.h5'

    # initialize model
    model = initialize_encoder(rows, columns, encoder_path)

    # pass the sets through the encoder
    ds_latent = model.predict(dataset)
    qs_latent = model.predict(queryset)

    # calculate the min & max values of the dataset latent vectors' values
    dsv_min, dsv_max = calculate_min_max(ds_latent)

    # produce the dataset and queryset output files
    produce_output_file(model, args.output_dataset, ds_latent, dsv_min, dsv_max, 25500)
    produce_output_file(model, args.output_queryset, qs_latent, dsv_min, dsv_max, 25500)

if __name__ == "__main__":
    """ call main() function here """
    print()
    # configure the level of the logging and the format of the messages
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s\n")
    # parse the command line input
    args = parse_input()
    # call the main() driver function
    main(args)
    print("\n")

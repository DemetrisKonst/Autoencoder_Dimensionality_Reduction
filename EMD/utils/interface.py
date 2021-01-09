import argparse

def parse_input(arg=None, autoencoder=True):
    """ function used to parse the command line input of the autoencoder """

    # create the argument parser
    description = "Python script that compares the EMD and Manhattan metrics on the MNIST dataset"
    parser = argparse.ArgumentParser(description=description)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the dataset."
    parser.add_argument("-d", "--dataset", type=str, action="store", metavar="dataset_path",
                        required=True, help=help)

    # add an argument for the path of the queryset
    help = "The full/relative path to the file containing the queryset."
    parser.add_argument("-q", "--queryset", type=str, action="store", metavar="queryset_path",
                        required=True, help=help)

    # add an argument for the path of the dataset labels
    help = "The full/relative path to the file containing the dataset labels."
    parser.add_argument("-l1", "--datasetlabels", type=str, action="store", metavar="dataset_labels_path",
                        required=True, help=help)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the queryset labels."
    parser.add_argument("-l2", "--querysetlabels", type=str, action="store", metavar="queryset_labels_path",
                        required=True, help=help)

    # add an argument for the path of the dataset
    help = "The full/relative path to the output file."
    parser.add_argument("-o", "--output", type=str, action="store", metavar="output_path",
                        required=True, help=help)

    # parse the arguments and return the result
    return parser.parse_args(arg)

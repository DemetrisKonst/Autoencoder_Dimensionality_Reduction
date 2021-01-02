import argparse

def parse_input(arg=None):
    """ function used to parse the command line input of the reducer """

    # create the argument parser
    description = "Python script that reduces the dimensionality of the MNIST dataset" \
                  "through a previously trained encoder."
    parser = argparse.ArgumentParser(description=description)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the dataset."
    parser.add_argument("-d", "--dataset", type=str, action="store", metavar="dataset_path",
                        required=True, help=help)

    # add an argument for the path of the queryset
    help = "The full/relative path to the file containing the queryset."
    parser.add_argument("-q", "--queryset", type=str, action="store",
                        metavar="queryset_path", required=True, help=help)

    # add an argument for the path of the output dataset
    help = "The full/relative path to the file to write the produced dataset."
    parser.add_argument("-od", "--output_dataset", type=str, action="store",
                        metavar="output_dataset_path", required=True, help=help)

    # add an argument for the path of the output queryset
    help = "The full/relative path to the file to write the produced queryset."
    parser.add_argument("-oq", "--output_queryset", type=str, action="store",
                        metavar="output_queryset_path", required=True, help=help)

    # parse the arguments and return the result
    return parser.parse_args(arg)

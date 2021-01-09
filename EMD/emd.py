import logging
import sys

sys.path.append("./utils")

from interface import *
from files import *
from utils import *
from metrics import *

def main(args):
    """ main() driver function """
    # first make sure that the paths to the provided files are valid
    if filepath_is_not_valid(args.dataset):
        logging.error("The path {} is not a file. Aborting..".format(args.dataset))
        exit()

    if filepath_is_not_valid(args.queryset):
        logging.error("The path {} is not a file. Aborting..".format(args.queryset))
        exit()

    if filepath_is_not_valid(args.datasetlabels):
        logging.error("The path {} is not a file. Aborting..".format(args.datasetlabels))
        exit()

    if filepath_is_not_valid(args.querysetlabels):
        logging.error("The path {} is not a file. Aborting..".format(args.querysetlabels))
        exit()

    # then make sure that the path to the output file is accessible
    if not filepath_can_be_reached(args.output):
        logging.error("The path {} cannot be reached to create file. Aborting..".format(args.output))
        exit()

    # parse the sets given
    dataset = parse_dataset(args.dataset)
    queryset = parse_dataset(args.queryset)
    dlabels = parse_labelset(args.datasetlabels)
    qlabels = parse_labelset(args.querysetlabels)


    if (check_total_image_values(dataset, queryset) == False):
        dataset = normalize_set(dataset)
        queryset = normalize_set(queryset)

    N = 10

    manhattan_neigbors, manhattan_distances = kNN(dataset, queryset[:300], N, manhattan)
    # print(manhattan_neigbors)
    # print(manhattan_distances)

    manhattan_avg = evaluate(dlabels, qlabels, manhattan_neigbors, N)
    print(manhattan_avg)

    # correct_count = 0
    # for i in range(len(manhattan_neigbors)):
    #     qlabel = qlabels[i]
    #
    #     for neighbor in manhattan_neigbors[i]:
    #         nlabel = dlabels[neighbor]
    #         if nlabel == qlabel:
    #             correct_count += 1
    #
    #
    # print(correct_count, "/", len(manhattan_neigbors)*5)


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

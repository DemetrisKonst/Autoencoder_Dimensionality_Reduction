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

    # if the images dont sum up to the same number
    if (check_total_image_values(dataset, queryset) == False):
        # normalize them
        dataset = normalize_set(dataset)
        queryset = normalize_set(queryset)

        # and check if they now properly sum up to the same number
        print("Dataset Normalized: ", check_total_image_values(dataset, queryset))

    N = 10
    cluster_size = 7

    # convert the dataset/queryset into weights of clusters
    cw_ds = convert_to_cluster(dataset, cluster_size)
    cw_qs = convert_to_cluster(queryset[:1], cluster_size)
    # calculate the euclidean distances between pixels of a 28x28 image
    centroid_distances = calculate_distances(cluster_size)

    # find kNN for all query images using EMD metric
    neighbors, neighbor_distances = kNN(cw_ds, cw_qs, centroid_distances, N)

    # calculate the average correct search results of the EMD metric
    emd_avg = evaluate(dlabels, qlabels, neighbors, N)

    # print/write results
    print("Average Correct Search Results EMD: {}".format(emd_avg))
    append_to_file(args.output, emd_avg)

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

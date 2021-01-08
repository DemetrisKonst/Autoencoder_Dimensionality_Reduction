from utils import filepath_can_be_reached
from error_utils import *

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32

DEFAULT_OPTION = 0
DEFAULT_GRAPH_OPTION = 1

def get_epochs():
    """ function used to read the number of epochs perfomed in the training of the autoencoder """

    # get the number of epochs
    prompt = "\nGive the number of epochs (default = {}): "
    prompt = prompt.format(DEFAULT_EPOCHS)
    epochs = input(prompt)

    # make sure the user gives a legit input
    while epochs != "":

        # try to convert the input to an int
        try:
            epochs = int(epochs)
            # it must be a positive integer
            if epochs <= 0:
                raise ValueError
            # if we get here then the input is fine, so break
            break

        # catch error and try again
        except ValueError:
            print("The number of epochs must a positive integer. Please try again.")
            epochs = input(prompt)

    # check if the user wants to use the deault value
    if epochs == "":
        epochs = DEFAULT_EPOCHS

    # return the final value
    return epochs


def get_batch_size():
    """ function used to read the batch size used in the training of the autoencoder """

    # get the number of batch_size
    prompt = "\nGive the batch size (default = {}): "
    prompt = prompt.format(DEFAULT_BATCH_SIZE)
    batch_size = input(prompt)

    # make sure the user gives a legit input
    while batch_size != "":

        # try to convert the input to an int
        try:
            batch_size = int(batch_size)
            # it must be a positive integer
            if batch_size <= 0:
                raise ValueError
            # if we get here then the input is fine, so break
            break

        # catch error and try again
        except ValueError:
            print("The batch size must a positive integer. Please try again.")
            batch_size = input(prompt)

    # check if the user wants to use the deault value
    if batch_size == "":
        batch_size = DEFAULT_BATCH_SIZE

    # return the final value
    return batch_size


def get_option():
    """ Function used to extract an option (0, 1, 2, 3) from the user after an experiment has
        been completed """

    # ask the user what he would like to do now
    prompt = "\nThe following options are now available:\n\t0) Exit the program\n\t1) Repeat " \
             "experiment with different values for the hyperprameters\n\t2) Show graphs with the " \
             "results of all the experiments run so far\n\t3) Print the results\n\n" \
             "Enter your action (default = 0): "
    option = input(prompt)

    # make sure that the option given is legit
    while option != "" and option != "0" and option != "1" and option != "2" and option != "3":

        # ask again as the input was wrong
        print("Wrong input, the option selected should be one of the following: 0, 1, 2 or 3. " \
              "Please try again.")
        option = input(prompt)

    # check for the default value
    if option == "":
        option = DEFAULT_OPTION
    # else, convert it to an int
    else:
        option = int(option)

    # return it
    return option


def get_graph_option():
    """ Function used to extract from the user whether he wants to see a loss vs epochs graph of the
        current experiment, or a loss vs hyperprameters graph over all the experiments """

    # ask the user
    prompt = "\nThere are 2 options available regarding the graphs that can be shown:\n\t1) Show " \
             "the Loss vs Epochs graph of the current experiment\n\t2) Show the Loss vs " \
             "hyperprameters graph over all the experiments run so far\n\nEnter your option " \
             "(default = 1): "
    answer = input(prompt)

    # while the answer is invalid
    while answer != "" and answer != "1" and answer != "2":
        # ask again as the input was wrong
        print("Wrong input, the option selected should be one of the following: 1 or 2. " \
              "Please try again.")
        option = input(prompt)

    # check whether the user wants the default option
    if answer == "":
        answer = DEFAULT_GRAPH_OPTION
    # else convert the answer to an integer
    else:
        answer = int(answer)

    # return the answer as an integer
    return answer

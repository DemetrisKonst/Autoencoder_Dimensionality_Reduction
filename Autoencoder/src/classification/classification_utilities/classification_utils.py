import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Input, Dense, Flatten, LeakyReLU
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import random

import logging
import sys
sys.path.append("../../utils")

from utils import *


def create_classifier(rows, columns, encoder, units):
    """
    Function that, given the encoder part, creates a "Model"
    (Keras object) that represents a classifier.
    """

    # define the input
    input = Input(shape=(rows, columns, 1))
    x = input

    # pass the input through the encoder
    x = encoder(x)

    # flatten
    x = Flatten()(x)

    # pass then the result through two fully-connected layers
    x = Dense(units=units, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # create the model and return it
    classifier = Model(input, x, name="classifier")
    return classifier


def show_experiment_graph(history, history_ft):
    """ Function used to show the Loss vs Epochs graph of one experiment """

    # get the losses
    train_losses = []
    train_losses.append(history.history["categorical_crossentropy"])
    train_losses.append(history_ft.history["categorical_crossentropy"])

    val_losses = []
    val_losses.append(history.history["val_categorical_crossentropy"])
    val_losses.append(history_ft.history["val_categorical_crossentropy"])

    # plot the losses
    epochs = len(train_losses)
    plt.xticks(np.arange(0, epochs, 1), np.arange(1, epochs + 1, 1))
    plt.plot(train_losses, label="Train Loss", color="mediumblue")
    plt.plot(val_losses, label="Validation Loss", color="darkred")

    # define some more parameters
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_graphs(histories, configurations):
    """ Function used to plot the losses of a model, for each configuration (experiment) tried """

    # get the number of experiments performed
    experiments = len(histories)

    # get the last losses of each experiment
    train_losses = [history.history["categorical_crossentropy"][-1] for history in histories]
    val_losses = [history.history["val_categorical_crossentropy"][-1] for history in histories]

    # now fix the x labels to match every experiment
    xlabels = []
    # add a label for each configuration
    for configuration in configurations:
        # get the values for that configuration
        units, epochs, batch_size = configuration
        # define the string and append it
        xlabel = "Units: {}\nEpochs: {}\nBatch Size: {}\n".format(units, epochs, batch_size)
        xlabels.append(xlabel)

    # define the parameters of the plot
    plt.xticks(np.arange(experiments), xlabels)

    # plot the losses
    plt.plot(train_losses, label="Train Loss", color="mediumblue")
    plt.plot(val_losses, label="Validation Loss", color="darkred")

    # define some more parameters
    plt.xlabel("Runs")
    plt.ylabel("Losses")
    plt.legend()
    plt.show()


def show_results(classifier, X_test, Y_test):
    """ function used to print the results of a trained model (classification report etc.) """
    # apply the test set on the trained model
    Y_pred = classifier.predict(X_test)

    # calculate the loss
    cce = CategoricalCrossentropy()
    loss = cce(Y_test, Y_pred).numpy()
    print("Test Loss: ", loss, "\n")

    Y_pred = np.round(Y_pred, 0)
    Y_pred = Y_pred.astype(int)

    # undo the binarization ([0,0,0,1] -> 4 etc.)
    Y_pred_unbin = np.argmax(Y_pred, 1)
    Y_test_unbin = np.argmax(Y_test, 1)

    # calculate the accuracy
    accuracy = accuracy_score(Y_test_unbin, Y_pred_unbin)
    print("Test Accuracy: ", accuracy, "\n")

    # find the amount of correct and incorrect predictions
    true_accuracy = accuracy_score(Y_test_unbin, Y_pred_unbin, normalize=False)
    print("Found ", true_accuracy, " correct labels", "\n")
    print("Found ", (Y_test.shape[0]-true_accuracy), " incorrect labels", "\n")

    # build the classification report
    target_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]
    report = classification_report(Y_test_unbin, Y_pred_unbin, target_names=target_names)
    print(report, "\n")


    # find the indices of images where the model predicted correctly/incorrectly
    Y_result = (Y_test_unbin == Y_pred_unbin)
    indices_correct = np.argwhere(Y_result==True)
    indices_incorrect = np.argwhere(Y_result==False)

    # shuffle those indices
    np.random.shuffle(indices_correct)
    np.random.shuffle(indices_incorrect)

    # get the first 12 indices from both lists and plot the image, the prediction and the actual label
    images_correct = [X_test[idx][0] for idx in indices_correct[:12]]
    im_c_label = [np.argmax(Y_test, 1)[idx] for idx in indices_correct[:12]]
    im_c_pred = [np.argmax(Y_pred, 1)[idx] for idx in indices_correct[:12]]

    images_incorrect = [X_test[idx][0] for idx in indices_incorrect[:12]]
    im_inc_label = [np.argmax(Y_test, 1)[idx] for idx in indices_incorrect[:12]]
    im_inc_pred = [np.argmax(Y_pred, 1)[idx] for idx in indices_incorrect[:12]]


    plot_example_images("Correct Predictions", images_correct, im_c_label, im_c_pred)
    plot_example_images("Incorrect Predictions", images_incorrect, im_inc_label, im_inc_pred)

def separate_to_clusters(Y):
    clusters = [[] for i in range(10)]

    for i in range(len(Y)):
        clusters[Y[i]].append(i)

    return clusters

def produce_label_file(clusters, file_path):
    file = open(file_path, 'w+')

    for i in range(len(clusters)):
        file.write("CLUSTER-{} {{ size: {}".format(i+1, len(clusters[i])))

        for index in clusters[i]:
            file.write(", {}".format(index))

        file.write("}\n")

    file.close()

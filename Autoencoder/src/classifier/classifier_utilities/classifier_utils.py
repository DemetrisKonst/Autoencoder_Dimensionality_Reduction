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

# function which creates a classifier model
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

# function which groups labels into num_clusters clusters
def separate_to_clusters(Y, num_clusters):
    clusters = [[] for i in range(num_clusters)]

    for i in range(len(Y)):
        clusters[Y[i]].append(i)

    return clusters

# function which produces a txt file based on the labels organized by clusters provided
def produce_label_file(clusters, file_path):
    file = open(file_path, 'w+')

    for i in range(len(clusters)):
        file.write("CLUSTER-{} {{ size: {}".format(i+1, len(clusters[i])))

        for index in clusters[i]:
            file.write(", {}".format(index))

        file.write("}\n")

    file.close()

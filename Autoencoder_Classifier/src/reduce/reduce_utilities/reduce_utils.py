import math

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten

import sys

sys.path.append("../../utils")

from utils import *
from dont_open import *

# preprocessing step for the data
def preprocess(data, rows, columns):
    # add one more dimension to be compatible with keras
    data = data.reshape(-1, rows, columns, 1)
    # normalize to 0-255
    data = data / 255.

    return data

def initialize_encoder(rows, columns, encoder_path):
    # load the encoder part
    encoder = load_keras_model(encoder_path)

    # create a new keras model with an encoder layer at first and a flatten layer at the end
    input = Input(shape=(rows, columns, 1))
    x = input
    # encoder layer
    x = encoder(x)
    # flatten layer
    x = Flatten()(x)
    model = Model(input, x, name="Reduce")

    return model

# function which calculates the min & max values of a vector of latent vectors,
# used to normalize the values of the latent vector
def calculate_min_max(data):
    max = 0
    min = math.inf
    for vector in data:
        for value in vector:
            if (value > max):
                max = value
            elif (value < min):
                min = value

    return (min, max)

# function which produces an output file based on the given data
# the data are passed through the encoder model and are then normalized evenly
# on the 0-factor range
# finally, they are written to a file in binary form
def produce_output_file(encoder, file_path, data, min, max, factor):
    item_amount = len(data)
    latent_dim = encoder.output.shape[1]

    # open file
    file = open(file_path, 'w+b')

    # write the magic number and the number of images, rows and columns
    file.write(struct.pack(">II", get_random_easter_egg(), item_amount))
    file.write(struct.pack(">II", 1, latent_dim))

    for vector in data:
        # for each value inside the latent vectors
        for value in vector:
            # normalize it to 0-25500
            write_val = int(((value - min)/(max - min))*factor)
            if (write_val > factor):
                write_val = factor
            # write it to the file as an unsigned half-word
            file.write(struct.pack(">H", write_val))

    file.close()

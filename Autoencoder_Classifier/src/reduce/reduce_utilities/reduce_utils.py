import math

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten

import sys

sys.path.append("../../utils")

from utils import *

def preprocess(data, rows, columns):
    data = data.reshape(-1, rows, columns, 1)
    data = data / 255.

    return data

def initialize_encoder(rows, columns):
    encoder_path = './encoder/z16.h5'
    encoder = load_keras_model(encoder_path)

    input = Input(shape=(rows, columns, 1))
    x = input
    x = encoder(x)
    x = Flatten()(x)
    model = Model(input, x, name="Reduce")

    return model

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

def produce_output_file(encoder, file_path, data, min, max, factor):
    item_amount = len(data)
    latent_dim = encoder.output.shape[1]

    file = open(file_path, 'w+b')

    file.write(struct.pack(">II", 80085, item_amount))
    file.write(struct.pack(">II", 1, latent_dim))

    for vector in data:
        for value in vector:
            write_val = int(((value - min)/(max - min))*factor)
            file.write(struct.pack(">H", write_val))

    file.close()

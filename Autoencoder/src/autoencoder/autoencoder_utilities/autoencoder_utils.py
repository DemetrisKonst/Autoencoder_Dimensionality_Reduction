import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, LeakyReLU
from tensorflow.keras.models import Model


def create_encoder(rows, columns, conv_layers, kernel_sizes, filters, use_third_max_pooling=True,
                   use_leaky_relu=False, leaky_relu_alpha=0.15):
    """
    Function used to create the encoder part of the autoencoder. The architecture follows the
    following rules:

    1) Every Convolutional Layer is followed by a Batch Batch Normalization.

    2) The First 3 Convolutional Layers also have a Max Pooling at the end. The first 2 Max Pooling
       layers have a pool size of (2, 2), while the third (if used) has a pool size of (7, 7). If
       less than 3 Convolutional Layers are available, then fewer Max Pooling layers are used.

    3) The activation function used in the Convolutional Layers is ReLU, but it can be changed to
       using Leaky ReLU just by setting the parameter "use_leaky_relu" to True and picking an alpha.

    4) Padding in Convolutions is always "same", that is, the output image from the convolution has
       the same shape as the input. Only the Max Pooling layers reduce the dimension of the images.
    """

    # define the input
    input = Input(shape=(rows, columns, 1))
    x = input

    # place the Convolutional-BatchNormalization-[MaxPooling] sets of layers
    for layer in range(conv_layers):

        # determine whether to use ReLU or Leaky ReLU
        if not use_leaky_relu:
            x = Conv2D(filters=filters[layer], kernel_size=kernel_sizes[layer], activation="relu",
                       padding="same")(x)
        else:
            x = Conv2D(filters=filters[layer], kernel_size=kernel_sizes[layer], activation="linear",
                       padding="same")(x)
            x = LeakyReLU(alpha=leaky_relu_alpha)(x)

        # perform batch normalization
        x = BatchNormalization()(x)

        """
        # second convolution in the same "set" of layers
        if not use_leaky_relu:
            x = Conv2D(filters=filters[layer], kernel_size=kernel_sizes[layer], activation="relu",
                       padding="same")(x)
        else:
            x = Conv2D(filters=filters[layer], kernel_size=kernel_sizes[layer], activation="linear",
                       padding="same")(x)
            x = LeakyReLU(alpha=leaky_relu_alpha)(x)

        # second batch normalization in the same "set" of layers
        x = BatchNormalization()(x)
        """

        # if we are placing a the 3rd layer and MaxPooling should be placed, do it
        if layer == 2 and use_third_max_pooling:
            x = MaxPooling2D(pool_size=(7, 7))(x)
        # if we are in the placing of the first 2 layers, place also a MaxPooling
        if layer < 2:
            x = MaxPooling2D(pool_size=(2, 2))(x)

    # create the encoder part and return it
    encoder = Model(input, x, name="encoder")
    return encoder


def create_decoder(rows, columns, conv_layers, kernel_sizes, filters, use_third_max_pooling=True,
                   use_leaky_relu=False, leaky_relu_alpha=0.15):
    """
    Function used to create the decoder part of the Autoencoder. The architecture is basically the
    "mirrored" architecture of the encoder.
    """

    # define the input
    factor = min(conv_layers, 2) * 2
    if conv_layers >= 3 and use_third_max_pooling:
        factor *= 7
    input_rows = rows // factor
    input_columns = columns // factor
    input = Input(shape=(input_rows, input_columns, filters[-1]))
    x = input

    # place the Convolutional-BatchNormalization-[MaxPooling] sets of layers in a mirrored way
    last_layer = len(filters) - 1
    for layer in range(last_layer, -1, -1):

        # determine whether to use Leaky ReLU or normal ReLU
        if not use_leaky_relu:
            x = Conv2D(filters=filters[layer], kernel_size=kernel_sizes[layer], activation="relu",
                       padding="same")(x)
        else:
            x = Conv2D(filters=filters[-1], kernel_size=kernel_sizes[-1], activation="linear",
                       padding="same")(x)
            x = LeakyReLU(alpha=leaky_relu_alpha)(x)

        # add batch normalization
        x = BatchNormalization()(x)

        """
        # second deconvolution in the same "set" of layers
        if not use_leaky_relu:
            x = Conv2D(filters=filters[layer], kernel_size=kernel_sizes[layer], activation="relu",
                       padding="same")(x)
        else:
            x = Conv2D(filters=filters[-1], kernel_size=kernel_sizes[-1], activation="linear",
                       padding="same")(input)
            x = LeakyReLU(alpha=leaky_relu_alpha)(x)

        # second batch normalization in the same "set" of layers
        x = BatchNormalization()(x)
        """

        # if we are in the third layer and a 3rd Max Pooling was placed, place an UpSampling
        if layer == 2 and use_third_max_pooling:
            x = UpSampling2D(size=(7, 7))(x)
        # if we are in the first 2 layers, place an UpSampling
        if layer < 2:
            x = UpSampling2D(size=(2, 2))(x)

    # do the final convolution to convert the image to the normal shape
    x = Conv2D(filters=1, kernel_size=kernel_sizes[-1], activation="sigmoid", padding="same")(x)

    # create the decoder part and return it
    decoder = Model(input, x, name="decoder")
    return decoder


def create_autoencoder(rows, columns, encoder, decoder):
    """
    Function that given the encoder part and the decoder part of an autoencoder, creates a "Model"
    (Keras object) that represents an autoencoder.
    """

    # define the input
    input = Input(shape=(rows, columns, 1))

    # pass the input through the encoder
    x = encoder(input)

    # pass then the result through the decoder
    x = decoder(x)

    # create the model and return it
    autoencoder = Model(input, x, name="autoencoder")
    return autoencoder


def show_experiment_graph(history):
    """ Function used to show the Loss vs Epochs graph of one experiment """

    # get the losses
    train_losses = history.history["mse"]
    val_losses = history.history["val_mse"]

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
    train_losses = [history.history["mse"][-1] for history in histories]
    val_losses = [history.history["val_mse"][-1] for history in histories]

    # now fix the x labels to match every experiment
    xlabels = []
    # add a label for each configuration
    for configuration in configurations:
        # get the values for that configuration
        conv_layers, kernel_sizes, filters, epochs, batch_size, third_maxpool = configuration
        # define the string and append it
        xlabel = "Conv Layers: {}\nKernels: {}\nFilters: {}\nEpochs: {}\nBatch Size: {}\n" \
                 "Third MaxPool: {}".format(conv_layers, kernel_sizes, filters, epochs, batch_size,
                                            third_maxpool)
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

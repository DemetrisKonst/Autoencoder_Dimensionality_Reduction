import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import logging
import sys
sys.path.append("../utils")
sys.path.append("classification_utilities")

from utils import *
from interface_utils import *
from classification_utils import *
from classification_interface_utils import *

def main(args):
    # first make sure that the paths to the provided dataset are valid
    if filepath_is_not_valid(args.data):
        logging.error("The path {} is not a file. Aborting..".format(args.data))
        exit()

    if filepath_is_not_valid(args.datalabels):
        logging.error("The path {} is not a file. Aborting..".format(args.datalabels))
        exit()

    if not filepath_can_be_reached(args.output_path):
        logging.error("The path {} is not a file. Aborting..".format(args.output_path))
        exit()

    if filepath_is_not_valid(args.model_path):
        logging.error("The path {} is not a file. Aborting..".format(args.model_path))
        exit()

    # parse the data from the training and test set
    X = parse_dataset(args.data)
    Y = parse_labelset(args.datalabels)

    rows = X.shape[1]
    columns = X.shape[2]

    # We also need to convert the labels to binary arrays
    lb = LabelBinarizer()
    Y = lb.fit_transform(Y)

    # reshape so that the shapes are (number_of_images, rows, columns, 1)
    X = X.reshape(-1, rows, columns, 1)

    # normalize
    X = X / 255.


    # split data to training and validation
    rs = 13
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=rs, shuffle=True)


    units, epochs, batch_size = (128, 1, 180)

    # load the encoder
    encoder = load_keras_model(args.model_path)
    # "freeze" its weights
    encoder.trainable = False

    # create the classifier using the encoder
    classifier = create_classifier(rows, columns, encoder, units)
    print()
    classifier.summary()

    # setup the classifier
    callback = ReduceLROnPlateau(monitor="val_loss", factor=1.0/2, patience=4, min_delta=0.005,
                                  cooldown=0, min_lr=1e-8, verbose=1)

    classifier.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy"])

    # train with the encoder frozen
    history = classifier.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                              shuffle=True, validation_data=(X_val, Y_val),
                              callbacks=[callback])


    # "unfreeze" the encoder
    encoder.trainable = True

    # now train the whole model
    history_ft = classifier.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                              shuffle=True, validation_data=(X_val, Y_val),
                              callbacks=[callback])

    Y_prob = classifier.predict(X)
    Y_pred = np.round(Y_prob)
    Y_unbin = np.argmax(Y_pred, 1)
    produce_label_file(Y_unbin, args.output_path)
    Yy = parse_labelset(args.output_path)
    print(Yy[10:20])


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

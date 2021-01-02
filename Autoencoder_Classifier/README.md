# Autoencoder for MNIST

## Team Members:
1. Andreas Spanopoulos (sdi1700146@di.uoa.gr)
2. Dimitrios Konstantinidis (sdi1700065@di.uoa.gr)
<br> </br>

## A little bit about our work
In this project we implement an Autoencoder for the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and a classifier that uses the encoder part of the autoencoder to reduce the dimensionality of images. Specifically, the project is broken down in two parts:
1. Create, train a Convolutional Autoencoder and save it.
2. Load the encoder part of the autoencoder and use it to create a classifier.

The project is implemented in Python, using the [Keras](https://keras.io/) library
for Deep Learning.
<br> </br>

## Autoencoder
The Autoencoder, which is implemented [here](src/autoencoder/autoencoder_utilities/autoencoder_utils.py), consists of 2 parts:
1. The Encoder
2. The Decoder

As their names suggest, the first has the job of "encoding" the input data into a
lower dimension, while the second "decodes" the compressed data into the original
dimension. How is this achieved? Let's take a look below

### Encoder
The encoder consists of "sets" of [Convolutional](https://keras.io/api/layers/convolution_layers/convolution2d/)-[BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/)-[MaxPooling](https://keras.io/api/layers/pooling_layers/max_pooling2d/) layers stacked together. Specifically, the rules for building the encoder are:

1. Every "set" of layers consists of a Convolutional layer and a Batch Normalization layer.
2. The first 2 "sets" of layers also have a MaxPooling layer after the Convolutional and Batch Normalization layers, with pool size equal to (2, 2).
3. The third "set" of layers, if specified by the user, will also have a MaxPooling layer in the end with pool size = (7, 7).
4. Padding is always "same", that is, the dimension of the output images from the convolutions have the same dimension as the input image.

In general, as we will also see below, the user picks the hyperprameters during the creation of the model, so he gets to choose how many Convolutional layers to add, their kernel sizes and filters, etc..

### Decoder
The decoder consists of "sets" of [Convolutional](https://keras.io/api/layers/convolution_layers/convolution2d/)-[BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/)-[UpSampling](https://keras.io/api/layers/reshaping_layers/up_sampling2d/) layers stacked together. The Decoder is always a "mirrored" representation of the encoder. For the sake of symmetry, let's also define the rules for it below:

1. Every "set" of layers consists of a Convolutional layer and a Batch Normalization layer.
2. The last 2 "sets" of layers also have an UpSampling layer after the Convolutional and Batch Normalization layers, with size equal to (2, 2).
3. The third-from-the-end "set" of layers, if specified by the user, will also have an UpSampling layer in the end with size equal to (7, 7).
4. Padding is always "same", that is, the dimension of the output images from the convolutions have the same dimension as the input image.
<br> </br>

## Classifier
The 2nd part of this project is a classifier. This classifier consists of two parts, the encoder and the [Fully-Connected](https://keras.io/api/layers/core_layers/dense/) layers. The encoder is the same that is generated from the 1st part. When the 1st part is executed, there is an option to save the encoder model with its weights and then migrate it to the classifier. The second part consists of the following:
1. One [Flatten](https://keras.io/api/layers/reshaping_layers/flatten/) layer which takes in the output of the encoder and formats it to be used in a fully-connected layer
2. One fully-connected layer which takes in as an input the flattened encoder output and the number of units used are determined by the user
3. One final fully-connected which takes in as input the output of the previous fully-connected and the output is 10 units (1 for each class)

##### Fine-Tuning
The training is split in two steps. First, only the fully-connected parts are trained, then the whole model is trained. This is achieved by [Freezing](https://keras.io/guides/transfer_learning/) the encoder model's weights. In essence, the model is trained 2 * epochs times.

## Benchmarks
Benchmarks for each part of the project can be found in the [benchmarks](benchmarks) directory.
<br> </br>


## Running
1. To run the autoencoder, first download the MNIST Dataset from [here](http://yann.lecun.com/exdb/mnist/), then go to the [src/autoencoder](src/autoencoder) directory and type the command
    ```bash
    $ python3 autoencoder.py -d ../../Dataset/train-images-idx3-ubyte
    ```
    in the terminal, where of course the path of the Dataset here will be substituted with the path on your machine.

    After the script starts running, the user will be asked to hand pick the hyperparameters that define the full architecture of the Network.
    
    In case the user wants to avoid hand-typing them every time, they can be stored in an input file, like this [one](src/autoencoder/input.txt), and be fed to the script with file redirection, as follows:
    ```bash
    $ python3 autoencoder.py -d ../../Dataset/train-images-idx3-ubyte < input.txt
    ```
2. To run the classifier, it is necessary to have an encoder model saved from the autoencoder executable. Assuming that, the user will already have downloaded the MNIST Dataset. Afterwards, you will have to navigate to the [src/classification](src/classification) directory and type the following command:
    ```bash
    $ python3 classification.py -d <training_set path> -dl <training_labels path> -t <test_set path> -tl <test_labels path> -model <encoder path>
    ```
    
    The user will then be prompted to tweak the hyperparameters of the model. Analytically, the number of units in the hidden fully-connected layer, the number of epochs and the batch size.


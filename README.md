# Project 3: *Autoencoder dimensionality reduction, EMD-Manhattan metrics comparison and classifier based clustering on MNIST dataset*

## Team Members:
1. Dimitrios Konstantinidis (sdi1700065@di.uoa.gr)
2. Andreas Spanopoulos (sdi1700146@di.uoa.gr)
<br> </br>

## A little bit about our work
In this project we implement 4 different evaluations regarding dimensionality reduction from an autoencoder model, comparison in NN search between
original item space, reduced item space and LSH application on the original space, comparison between the Wasserstein and Manhattan metrics on NN, comparison
between centroid-based clustering through k-medians with classifier based clustering.

1. The program reduce.py accepts as input a dataset and using a pretrained encoder produces a new dataset based on the latent vector output of the encoder.
2. The program search.cpp accepts the original dataset input and the modified input from reduce.py and executes a comparison between Brute Force NN search in
the original space, Brute Force NN search in the reduced space and LSH ANN search in the original space.
3. **TODO**
4. The program cluster.cpp accepts the original dataset input, the modified input from reduce.py and the predictions made by classification.py to produce 3
different cluster models, afterwards it calculates some valuable metrics on each to evaluate their performance.

## A little bit about our Dataset
The Datasets used to test the correctness of our algorithms is the *MNIST Database of handwritten digits*, and can be found [here](http://yann.lecun.com/exdb/mnist/). This database contains images of size 28 x 28 pixels, that is, 784 pixels if we flatten the image (spoiler alert: we do). The Training set consists of 60000 images, and the test/query set of 10000. Note that the images are stored in Big-Endian format. Our code makes sure that they are converted to Little-Endian.
<br> </br>

## Part 1 - Dimensionality Reduction
As stated before, this program converts a given dataset to a reduced version of it by passing it through a pretrained encoder model and taking the latent vector values produced by it. Analytically, the first part of this program is to create a viable encoder model, thankfully, the [autoencoder](Autoencoder_Classifier/src/autencoder) directory contains a program which trains an autoencoder on a given dataset and allows the user to save the encoder part of the model. Hence, we would strongly advise the encoder used to be produced from this specific program (Note: the [notebook](Autoencoder_Classifier/notebook) directory contains an almost identical colab notebook implementation of the autoencoder for easier GPU usage). After the encoder model has been saved, we are free to reduce the dimensionality of our dataset. The reduce.py file contains a local variable pointing to the encoder to-be-used which can be changed to specify the user's preference. Essentially, the dataset and queryset given as arguments are passed through the encoder model, flattened and then written to the respective output files (given as arguments). **More on usage later**.

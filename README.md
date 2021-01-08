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
the original space, Brute Force NN search in the reduced space and LSH NN search in the original space.
3. **TODO**
4. The program cluster.cpp accepts the original dataset input, the modified input from reduce.py and the predictions made by classification.py to produce 3
different cluster models, afterwards it calculates some valuable metrics on each to evaluate their performance.

## A little bit about our Dataset
The Datasets used to test the correctness of our algorithms is the *MNIST Database of handwritten digits*, and be found [here](http://yann.lecun.com/exdb/mnist/). 
This database contains images of size 28 x 28 pixels, that is, 784 pixels if we flatten the image (spoiler alert: we do). 
The Training set consists of 60000 images, and the test/query set of 10000. 
Note that the images are stored in Big-Endian format. Our code makes sure that they are converted to Little-Endian.
<br> </br>

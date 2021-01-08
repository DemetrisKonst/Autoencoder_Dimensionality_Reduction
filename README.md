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
As stated before, this program converts a given dataset to a reduced version of it by passing it through a pretrained encoder model and taking the latent vector values produced by it. Analytically, the first part of this program is to create a viable encoder model, thankfully, the [autoencoder](Autoencoder/src/autencoder) directory contains a program which trains an autoencoder on a given dataset and allows the user to save the encoder part of the model. Hence, we would strongly advise the encoder used to be produced from this specific program (Note: the [notebook](Autoencoder/notebook) directory contains an almost identical colab notebook implementation of the autoencoder for easier GPU usage). After the encoder model has been saved, we are free to reduce the dimensionality of our dataset. The reduce.py file contains a local variable pointing to the encoder to-be-used which can be changed to specify the user's preference. Essentially, the dataset and queryset given as arguments are passed through the encoder model, flattened and then written to the respective output files (given as arguments).

## Part 2 - Original Space & Latent Space brute force NN and LSH ANN on original space comparison
This program accepts as input a dataset and a queryset both in the original and the reduced dimension versions of them (i.e. the output of **Part 1**). Then makes comparisons between:
- Brute Force NN on original space.
- Brute Force NN on reduced space.
- LSH ANN on original space.
With respect to the brute force NN on original space, approximation factors of the latter 2 approaches are calculated (based on relative distance of neighbor produced). Essentially, the original dataset is loaded into both a [BruteForce](NN_Clustering/include/BruteForce/BruteForce.hpp) class and an [LSH](NN_Clustering/include/LSH/LSH.hpp) class, alongside them, the reduced dataset is loaded into a [BruteForce](NN_Clustering/include/BruteForce/BruteForce.hpp) class. Afterwards, for each query image of the respective queryset (original/reduced) the Nearest Neighbor is calculated on all 3 models. Finally, all these calculations are written to an output file where some extra information is recorded (approximation factors, times etc.).

## Part 3 - Earth Mover's Distance & Manhattan Distance metrics comparison
**TODO**

## Part 4 - Centroid-based clustering on original & latent space and classifier based clustering comparison
The program that implements this comparison is [cluster.cpp](NN_clustering/cluster.cpp). It accepts as arguments the dataset in the original space, in latent space, a configuration file (containing parameters regarding the clustering), a path to the output file and a file containing grouped images per label from the dataset produced by a classifier. We have implemented a [classifier](Autoencoder/src/classifier/classifier.py) like such that based on a pre-trained encoder classifies the images of the dataset based on the digits they represent. This file's format is:
```
CLUSTER-label {size: amount_of_image_ids, image_id_1, image_id_2, ..., image_id_n}
CLUSTER-next_label {....}
```
You are free to use your own file (as long as it complies with the specified format). To use our classifier, at first you need to have a pre-trained encoder saved. The one created by the [autoencoder](Autencoder/src/autoencoder/autoencoder.py) used in **Part 1** works perfectly.
## Usage
### Part 1
To execute the autoencoder.py program (to produce the encoder), type:
```bash
  $ python3 autencoder.py -d dataset
```
You will then be prompted for the location of the encoder model to be saved at.
There is also the option to use the colab notebook in the [notebook](Autoencoder/notebook) directory.

To specify the encoder in the reduce.py program, change the local variable named "encoder_path" with the path of the encoder you want to use.
To execute:
```bash
  $ python3 reduce.py -d dataset -q queryset -od output_dataset_file -oq output_query_file
```
Then, the dataset & queryset produced files will be created to be used for **Part 2**

### Part 2
To execute the search program, first make sure to use
```bash
  $ make SEARCH
```
inside the [NN_Clustering](NN_Clustering) directory. Afterwards, use the following command to run the program
```bash
  $ bin/search  –d input_file_original_space
                -i input_file_new_space
                –q query_file_original_space
                -s query_file_new_space
                –k number_of_LSH_hash_functions
                -L number_of_LSH_hash_tables
                -ο output_file
```
Then, the file containing information about the experiment will be produced to the path specified.

### Part 3
**TODO**

### Part 4
To execute the classifier so as to produce the cluster file, type:
```bash
  $ python3 classifier.py -d dataset -dl dataset_labels -ol output_file -model encoder
```
inside the [classifier directory](Autencoder/src/classifier). Afterwards, the cluster file will have been produced in the output_path specified and we are free to execute the clustering program.
First, make sure to use
```bash
  $ make CLUSTER
```
inside the [NN_Clustering](NN_Clustering) directory. Afterwards, use the following command to run the program
```bash
  $ bin/cluster  –d input_file_original_space
                -i input_file_new_space
                –n cluster_file
                -c configuration_file
                -ο output_file
```
Then, the file containing information about the experiment will be produced to the path specified.

### *[GitHub Repository](https://github.com/DemetrisKonst/Autoencoder_Dimensionality_Reduction)*

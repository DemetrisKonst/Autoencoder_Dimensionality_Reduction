# Analysis
In the current file the results of all algorithms regarding the 4 parts of this project will be analyzed with multiple experiments at a time.

## Part 1
The first parameter that can be tailored is the architecture of the encoder. The [autoencoder](Autoencoder/src/autoencoder/autoencoder.py) file and its [notebook](Autoencoder/notebook/Autoencoder.ipynb) counterpart support many different configuration options like number of convolutional layers, kernel sizes etc. In order to achieve a small dimension for the latent vector, the filter sizes have to be small and the usage of a third max-pooling layer is necessary. The following graph shows the MSE loss among 4 different experiments with latent vector dimensions of 2, 4, 8 and 10 respectively.


![image](./images/autoencoders.png)

The 2nd experiment (with latent dimension of 4) seems the most promising one so this is the one we will be using to generate the new space for **Part 2**.

## Part 2
Having used the 2nd experiment mentioned above (latent dimension = 4) we executed the [search](NN_Clustering/search.cpp) program. For the LSH parameters we used k = 4 and L = 4 as they are a good all-around combination (more on that explained in this [respository](https://github.com/DemetrisKonst/ANN-Clustering) we have created for LSH). The results were the following:
<br><br>
- Time-related results
    - Reduced Total Time: 6.27314
    - LSH Total Time: 7.81783
    - True Total Time: 47.5043
- Accuracy-related results
    - Approximation Factor Reduced: 1.89546
    - Approximation Factor LSH: 1.06049
<br><br>
First of all, it is clear that both the LSH run and the Reduced space run were much faster than the Original space run. Among the two, the reduced approach seems to be faster but not by a huge amount. Accuracy-wise, LSH is performing as expected, nearing 100% accuracy. On the other hand, the brute force approach on the reduced space is disappointing to say the least, provided that we used the "best" encoder we had.
<br><br>
**Note**: We could have obviously used an encoder of a huge latent dimension (e.g. 512) to reach way better results on the reduced space, however that would miss the point of these experiments.

## Part 3
The experiments for this part are pretty straight-forward. The [manhattan](NN_Clustering/manhattan.cpp) evaluation does not accept any parameters, so there is only need to run one single experiment. At this run, the result was the following:
- Average Correct Search Results MANHATTAN: 0.93826
<br><br>
Regarding the EMD metric, there is one parameter that can be tweaked: the size of the clusters of the photo. As our images are 28x28, the size has to be either 4x4 or 7x7. However, only a size of 7x7 is actually viable for evaluation, as the 4x4 cluster size does not reduce the amount of weights sufficiently enough, so the program is **very** slow. So, for a cluster size of 7x7 we had the following results:
<br><br>

| Dataset Size | Queryset size | Accuracy | Time |
| ------------ | ------------- | -------- | ---- |
| 10000        |  10           |  0.68    | 13m  |
| 60000        |  10           |  0.80    | 77m  |
| 10000        |  60           |  0.73    | 76m  |

<br><br>
The evaluation of the EMD metric -as expected- is very costly timewise, if we were to run the whole queryset (10k images) on the whole dataset (60k images), the time to do all that would be approximately 1 month and 24 days, so we obviously ran tests on subsets of those. Whether the actual accuracy of the EMD metric is around 0.8 (based on the 2nd experiment), we can't be sure unless we use some hypercomputers. However, it seems improbable that it will surpass the 0.93 mark of the Manhattan metric.

## Part 4
To classify the images into classes, we used a very powerful classifier which has an accuracy on the test set of around 99%, no more needs be said about this as there is extensive experimentation regarding the different classifier models in this [repository](https://github.com/AndrewSpano/Autoencoder_for_MNIST). 
<br>
Having used that, we ran the [cluster](NN_Clustering/cluster.cpp) on the same reduced space used for **Part 2** mentioned above. The list of all silhouettes for each cluster would be too long to include here, so we will keep just the objective function and the average silhouette for each model.
<br><br>

| Model               | Value of Objective Function | Average Silhouette |
| ------------------- | --------------------------- | ------------------ |
| Original Space      | 1.1547e+09                  | 0.0858673          |
| Reduced Space       | 1.49471e+09                 | -0.00315426        |
| Classes as clusters | 1.22244e+09                 | 0.0971437          |
<br><br>

Technically speaking, assigning images of the same classes to the same clusters, should produce an optimal clustering. This is what happens, as we can see that the specific configuration achieves the highest average silhouette score. The clustering on the original space has a lower silhouette value, but also a much lower objective function value. This is to show that an optimal clustering does not always imply a global minima in the objective function, as can be seen in the example below.
<br><br>

<p align="center">
  <img src="./images/mickey.png" alt="Sublime's custom image"/>
</p>

<br><br>
The clustering in the reduced space is much faster, but of course does not achieve an acceptable performance. This is due to the big information loss from the original 28x28 pixels of the images, to a latent vector of size 4. This behavious is expected, and can be hardly improven.

We conclude that using the classes of their images as their clusters, achieves better performance and is much faster. The downside is that in reality, when dealing with unsupervised learning we do not have labels of the input. Therefore, this technique can't be applied in the real world. Lloyd's (brute-force) clustering yields decent results, but is much slower, especially in the Big Data era. Performing clustering in the reduced space will improve running time, but decrease performance. One could argue that the tradeoff is slightly imblanaced, and that it is not worth to sacrifice all this computing power, for such worse results.
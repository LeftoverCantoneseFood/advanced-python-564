1) Unsupervised learning is when our data is unlabelled, and we ask the algorithm to learn something meaningful on its own without labels.

2) The 2 types of unsupervised learning we will explore today are:
Principal Component Analysis (PCA)
K-Means Clustering


3) What is a centroid? Take some time on this question after reading under the K-Means block in the ipynb. Draw some of your understanding out on the board! Include some formulas if you can.
The centroid is the geometric mean (center, as in centroid) of all the points in a cluster.

4) Under line 7, the blue points represent all datapoints (not assigned to centroids) and the stars represent the centroids of each (future) cluster.
OHHHH this says line 7

5) Write the formula for finding the closest centroids on the board, and then summarize it using your own words here.
c(i) = j that minimizes || x(i) - μ(j) ||
closest centroid = j such that magnitude of xPosition - centroidPosition (of centroid j) is as small as possible.

6) Read the code in line 8. Name the type of python object each of these are:

centroids, X, np.square, min_distance, idx
numpy array, numpy array, function (numpy), integer/float, function (pandas)

7) What is the purpose of line 9?
Find the closest centroid for each point given to the function

8) What is the difference between the first plot under line 7 versus the one in line 10?
Points have been assigned and recolored based on their closest centoroid. Each different color of points represents a new cluster. Clusters are determined based on what the closest centroid to each point is.

9) On the board, write the equation under 'Computing Centroid Means' and label each of the variables. On your canvas submission, write 2 sentences on what this equation represents.
Add the position of each datapoint within the cluster k. Then divide it by the number of datapoints in the cluster k. Then that is the position of centroid(k). This equation finds the weighted average of all the points in the cluster, where farther points matter less.

10) Challenge: What is the purpose of line 17-24?
17 - train and fit KMeans model on given data
18 - declare x and y for a plot
19 - plot the datapoints and their centroids, show plot
20 - initialize centroids by picking points at random
21 - print coordinates of centroids
24 - iterate 3 times to move centroids closer to where they should be

11) Line 25 onwards shows you how to do exactly what the above lines did with hard code. That means that scikit learn has an inbuilt function for all the things we just hand-coded! Yay! Write the code that helps you do what we did faster here:

from sklearn.cluster import KMeans

km1= KMeans(3)

km1.fit (X1)

12) Look up the parameters for Kmeans and explain them here!
n_clusters - how many clusters (and therefore centroids) to generate.
init - method for initialization
    random - place centroids at random datapoints
    k-means++ (defalt) - pick first centroid randomly, and then weigh the selection so that farther-away data points are more likely to be picked in subsequent rounds

13) Check out the image compression example from lines 27-36, and write at least 3 questions on the board that you have about it (if you don't have any questions, write 3 'takeaways'.
    - CV models can't actually see or look at images the way we do.
    - we have to represent images in a way that computers can understand, preferably with numerical values that allow us to do data processing & run analysis/algorithms
    - we can represent each image as an array of pixels. we can then encode pixels using RGB values, representing pixels as an array of RGB values.
    - these RGB arrays are what we feed into the model for training
    - we can simplify images by reducing the amount of colors. in this exercise we are reducing to 16 colors, and using 4 bits to represent one of thw 16 colors, reducing our memory burden of storing the image.
    - in this compression model, we are going to use K-means clustering to find the 16 most optimal colors to represent the compressed image.
    - Each pixel will belong to one of 16 clusters. The mean of the clusters, the centroid, will become the new color for all pixels in the cluster when the image is compressed.

14) It's PCA time. What type of object is X3 in line 39?
    - it's an array that's 50 entries, and 2 columns per entry

15) What type of normalization are they doing for this data?
    - compute the mean of the data
    - subtract this mean from the data
    - compute the standard deviation post-subtraction
    - divide the post-subtraction data by the standard deviation

16) Vectors are kinda hard. Find a resource online to help you understand #3-- I'm here to help! Insert the resource here.
    - a vector is essentially an array
    - an eigenvector is the array

17) In line 42, what is the purpose of this function, and what type of objects is the function returning?

18) Don't worry about all the particularities of line 47, data visualization is it's own class and specialty, but check out the graph. Read the paragraph just below and explain it in a sentence in normal-people speak here. I would ignore the paragraph after that unless you're feeling brave.

19) Skip ahead to the graph under line 54. Write a sentence about what this is doing for me.

20) Lines 55 do the thing that happened in the above lines for PCA with an in-built function in scikit learn. Insert the code here:

from _______________ import ________

pca= ___________ (n_components= ____)

pca_result= pca.fit_transform(____)

21) In the code you just wrote above, where did X3 even come from? What is it? And what is n_components anyway?


22) Somewhere before the graph are a few sentences that I would like you to comprehend:

pca_result: (50, 2)
Variation per principal component: [0.87062385 0.12937615]
The above ratio is how much of the variance or variability in our data is captured by the two principal components.

You may have to look this up, but what does the (50, 2) mean? And in normal people speak, what does the output for Variation per principal component mean?

23) Skip ahead to line 89, and read through the code, it's kinda cool!

24) Here's the fun now... Import the important libraries you need in a fresh ipynb, and run this code:

from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

china = load_sample_image("china.jpg")
plt.imshow(china)
plt.axis('off')

Can you try to compress this image for me, using K-Means?

25) Extra super bonus if you are already done: Find a way to upload your own (class appropriate) image into an ipynb, and see if you can reduce the number of colors in it.

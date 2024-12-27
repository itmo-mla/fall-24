
### **Lab #1 Clustering**

#### <p style='text-aligh: justify'> 1. For this lab work I have selected the [IRIS Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset) and [Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data) datasets from kaggle. </p>
####   <p style='text-aligh: justify'> 1.1 Since both datasets didn't have class labels there was no need to perform the first step in the given tasks. Although FinalGrade from the Student Performance can be considered as a class label I decided to treat it as a feature. </p>

#### 1.2 I have used scatter plots of the given datasets to visualise the data in the datasets. I use the [importing_data.py](./source/importing_data.py) to import and visualise the data in one go.

<img src="img/image1.png">

####  <p style='text-aligh: justify'> 1.3 The first dataset is made up of **clusters with jumpers**. As clearly seen on the plot we have one or two points that connect the areas with dense consentration of points. As for the second dataset, it has no clear shape but I assume is made up of **overlapping clusters**.  </p>

#### <p style='text-aligh: justify'> 1.4 Though it is not clear, for the first dataset my assumption is at least 5 clusters by looking at the graph. As for the second dataset, there seem to be 2 or 3 areas with consentration but I cannot be so sure.</p>

#### 2. I have implented a hierarchical algorithm based on Agglomerative clustering. [my_hierarchy.py](./source/my_hierarchy.py) contains the code for this implementation. It takes the number of desired clusters, linkage method and a dataframe as arguements. It outputs the linkage matrix and labels. I have implemented a function tha takes these and plots a dendrogram.

<img src="img/image2.png">
<img src="img/image3.png">

#### 2.1 Dendrograms for the two datasets
Dendrogram plotted using the implemented hierarchical algorithm.

<img src="img/image4.png">

Dendrogram using the reference algorithm.

<img src="img/image5.png">

As we can see in the figures, we have similar results.
        
#### 2.2 Optimum number of Clusters

#### <p style='text-aligh: justify'> The optimum number of clusters is calculated using maximum increment R_t for hierarchical clustering based on Ward distance. </p>

The results on the first dataset
```
Optimum number of clusters for dataset 1 : 4
Max increment distance: 1668.6530013777597
```
The results on the second dataset
```
Optimum number of clusters for dataset 2 : 4
Max increment distance: 29.204787217021345
```
    
####    According to the results for the Mall Customers Dataset 4 clusters seems to be the appropriate number. For the Iris dataset, I chose also 4 clusters as per the results. 

#### 3. I have implemented [EM algorithm](./source/em.py) and [DBSCAN algorithm](./source/dbscan.py).

<img src="img/image6.png" width='400'>
<img src="img/image7.png" width='400'>

<img src="img/image8.png">
<img src="img/image9.png">

#### 4. The following metrics were applied on those algorithms.
####    * Based on the clustering results, calculated the average intra-cluster distance;
####    * Based on the clustering results, calculated the average inter cluster distance;
####    * measured the clustering speed;

### For Mall Customers: 

| | Metric |	Implemented Algorithms |	Library Algorithm |
| - |----- | ----------------------- | ------------------ |
|1 |	Average Inter-Cluster Distance (Agglomerative) |	56.115664 |	58.271701 |
| 2 |	Average Inter-Cluster Distance (EM) |	52.191699 |	56.166705 |
| 3 |	Average Inter-Cluster Distance (DBSCAN) |	53.652448 |	46.076062 |
| 4 |	Average Intra-Cluster Distance (Agglomerative) |	136.440442 |	123.414764 |
| 5 |	Average Intra-Cluster Distance (EM) |	34.838372	| 26.156562 |
| 6 |	Average Intra-Cluster Distance (DBSCAN) |	23.393297 |	46.076062 |
| 7 |	Clustering Speed (Agglomerative) |	3.202118 |	0.002134 |
| 8 |	Clustering Speed (EM) |	0.037432 |	0.046854 |
| 9 |	Clustering Speed (DBSCAN) |	0.000707 |	0.0.005861 |


### For IRIS Dataset: 

| | Metric |	Implemented Algorithms |	Library Algorithm |
| - |----- | ----------------------- | ------------------ |
|0|	Average Inter-Cluster Distance (Agglomerative)|	1.355990|	2.188484|
|1|	Average Inter-Cluster Distance (EM)|	2.740543|	0.586880|
|2|	Average Inter-Cluster Distance (DBSCAN)|	2.757433|	2.300989|
|3|	Average Intra-Cluster Distance (Agglomerative)|	2.691691|	2.976170|
|4|	Average Intra-Cluster Distance (EM)|	0.511860|	0.586880|
|5|	Average Intra-Cluster Distance (DBSCAN)|	0.813546|	2.300989|
|6|	Clustering Speed (Agglomerative)|	2.160613 |	0.002373|
|7|	Clustering Speed (EM)|	0.286408|	0.151743|
|8|	Clustering Speed (DBSCAN)|	0.000641|	0.013844|

#### 5. The following metrics were applied on reference algorithms.
####    * Based on the clustering results, calculated the average intra-cluster distance;
####    * Based on the clustering results, calculated the average inter cluster distance;
####    * measured the clustering speed;

#### 6. Comparisons between implemented and reference algorithms is the following.

<img src="img/image10.png">

#### <p> Average Inter-Cluster Distance: Lower values indicate better separation between clusters.
#### Conclusion: Implemented hierarchical clustering performs better on the second dataset and a little lower on the first. EM clustering performs almost the same as the library implementation on the first dataset but on the second dataset the library implementation performs a little bit better. My DBSCAN implementation performs closer to the library version on the first dataset but performs poorly in the second dataset.

#### Average Intra-Cluster Distance: Lower values indicate that points within the same cluster are closer together.
#### Conclusion: Both EM algorithms have almost the same average intra-cluster distance, indicating better compactness within clusters. However, the implemented DBSCAN algorithm has a much lower intra-cluster distance compared to the library version on both datasets. Implemented hierarchical algorithm perform poorly on both datasets.

#### Clustering Speed: Lower values indicate faster clustering.
#### Conclusion: Both implemented and library algorithms performed fast but the library algorithms are a little bit faster on almost all datasets.</p>


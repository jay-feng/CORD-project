# Clustering and Understanding COVID-19 Research Papers

### Introduction
This project is based on the Kaggle COVID-19 Open Research Dataset (CORD) Challenge, which asks participants to extract information from a dataset of 40,000+ research papers on COVID-19 in order to answer various questions regarding COVID-19. Because I was not sure which questions I would be able to answer with the data,  I started my research by clustering the documents and observing the different topics within the corpus. In this project I decided to search for relevant text regarding the question “What has been published about medical care?” but the clustering and text extraction techniques can be generalized to answer any of the other questions listed on the Kaggle page.

### Data Cleaning
The main data frame provided on the Kaggle page contained information about the article, including date of publication, author, abstract, whether it had a full text file available, and the license for each full text file. To make the texts of the papers more usable, I filtered out papers without a full text file, pulled the body paragraph texts out of the corresponding .json files of the remaining papers, and put them into a dictionary that mapped the paper IDs to the full text as a string. This enabled easy access to a string of the body text of each paper for important text extraction later on.

![error](https://github.com/jay-feng/CORD-project/blob/media/main-df.png?raw=true)

![error](https://github.com/jay-feng/CORD-project/blob/media/json-format.png?raw=true)

### Clustering - K-means
Because of the vast size of the full-text strings, I decided to just use the paper abstracts for clustering, under the assumption that the abstracts would contain the keywords of the paper. The first method I used was K-means, which is an unsupervised learning algorithm that assigns data points into “clusters” with nearby points. The model starts with k randomized centroids and assigns each data point to a centroid, adjusting the centroids to minimize the variance between points in the cluster. This requires the text to be vectorized, which I achieved by using sk-learn’s TF-IDF vectorizer, which assigns a score to each word that appears in the collection of papers based by how many times it occurs in the given paper and how often it appears in all of the papers. However, before applying TF-IDF, I cleaned the data by removing words with non a-z/A-Z characters with regex, filtering out stop words (“a”, “the”, “of”) which contribute little meaning, converting words to their stems (running -> run) so words with the same meaning would be grouped together, and taking the top 150 most significant words in the corpus to reduce dimensionality. Because each vector has a value for each word that appears in the entire collection of papers, the resulting set of vectors contains mostly 0 values and has a very high dimensionality.

![error](https://github.com/jay-feng/CORD-project/blob/media/tfidf-vectors.png?raw=true)

(There are 149 columns because the column “abstract” was removed.)

Because of the high dimensionality of each vector (representing a single document), the distance between points would be very large and shared words between documents would only marginally reduce that distance, thus decreasing the effectiveness of the k-means algorithm. To reduce the dimensionality of the data, I used PCA (principal component analysis), which finds a new orthogonal basis for the n dimensional vectors, where the data points have a high variance with an n-1 dimensional subspace and a low covariance with the orthogonal 1 dimensional basis vector. All the points are then projected onto the n-1 dimensional subspace, thus reducing the dimensionality of the data while maintaining most of its meaning. This is repeated until the desired number of dimensions, or principal components, are achieved. I used the sk-learn implementation of the algorithm with a variance threshold of 0.85, which would find the maximum dimensionality reduction while still maintaining 85% of the variance so the projections would not cause the data to lose substantial information. This brought the vectors down to about 110 dimensions.

To find the optimal amount of clusters for the dataset, I created an elbow plot of the sum of squared errors with k (number of clusters) values from 2 to 30.

![error](https://github.com/jay-feng/CORD-project/blob/media/kmeans-elbow-plot.png?raw=true)

Ideally, the SSE would decrease to a certain point until levelling out, and the point at which it levels out is the optimal number of clusters to reduce the distances between points in a cluster. This would show that a cluster is indeed a group of closely connected points. However, the plot does not clearly show that the SSE levels out at a certain number of clusters. This indicates that the data does not contain clear clusters using the calculated TF-IDF values, so increasing the number of clusters only makes each cluster smaller, thus naturally reducing the SSE, without making more meaningful clusters. Although not ideal, ~14 seems to be the optimal number of clusters from the graph.

Using t-SNE, another dimensionality reduction technique often used for visualization, I plotted the points in two dimensions with their assigned clusters.

![error](https://github.com/jay-feng/CORD-project/blob/media/kmeans-tsne-plot.png?raw=true)

While there are some well defined clusters, many of the points in the center of this graph seem like they could belong to many different clusters.

### Clustering - LDA (Latent Dirichlet Allocation)
Because k-means did not seem to be very effective, I decided to create a different set of clusters using LDA (Latent Dirichlet allocation) topic modeling, which is a bag of words model as opposed to k-means, a distance based algorithm. Similar to k-means, LDA assumes there exists k number of topics, each represented by a set of words. LDA represents each document as a mixture of topics, based on what topics the words in the document belong to. The model works by first assigning each word in each document to a random topic, and reassigning the topic for each word based on the probability that a topic t represents a document and the probability that the given word comes from that topic t. (A more detailed introduction to LDA can be found here.) While it was more difficult to find a metric like the SSE in k-means to measure how “meaningful” clusters are, it was easy to observe the words that represented each topic created by the LDA model.

![error](https://github.com/jay-feng/CORD-project/blob/media/lda-topic-examples.png?raw=true)

I tweaked the number of topics created by increasing the number of topics until they had overlapping words and themes, so I could ensure the model was finding all the topics in the corpus without creating meaninglessly overlapping topics. I found that ten topics was the most ideal. Each topic was represented by a set of word-probability paires, so I took the ten words with the highest corresponding probability values for each topic and assigned every document to a topic by finding the topic where the corresponding words appeared most often in the document. Here are word clouds of the most common words appearing in the documents of each topic.

![error](https://github.com/jay-feng/CORD-project/blob/media/word-clouds.png?raw=true)

Most importantly, here is the word cloud I thought best represented the medical care topic:

![error](https://github.com/jay-feng/CORD-project/blob/media/medical-word-cloud.png?raw=true)

Because these words occurred often in the documents with the medical care topic, they are understandably broad and tell us little about the individual papers themselves. However, the clustering helps to narrow down the search for papers regarding medical care, from which we can extract more specific insights.

### Important Text Extraction
To extract the most important sentences from the documents in the medical care topic, I split the text into sentences and applied the TF-IDF vectorizer on each sentence, which assigned a score to each word in the sentence based on how “important” that word was relative to the entire document. The sentences with the highest sum of TF-IDF scores would be the “most important” sentences in the document. However, many of the sentences with high TF-IDF sums were extremely long sentences, so I divided the TF-IDF sum by the log of the length of the sentence to reduce bias towards longer sentences. Here are a few examples:

![error](https://github.com/jay-feng/CORD-project/blob/media/important-sentences.png?raw=true)

### Conclusion
While I ended up extracting the “most important” sentences in the relevant documents, this was more of a proof of concept and a means to complete the task described in the Kaggle challenge. However, the results of the clustering/topic assignments could easily be adapted for more practical purposes, such as searching for papers with a desired topic or summarizing the papers in a certain topic to quickly identify which of the tens of thousands of papers are worth looking at in more detail.

### References
Abdullatif, H. (2020, January 28). Dimensionality Reduction For Dummies - Part 1: Intuition. Retrieved from https://towardsdatascience.com/https-medium-com-abdullatif-h-dimensionality-reduction-for-dummies-part-1-a8c9ec7b7e79

Abdullatif, H. (2020, March 24). You Don't Know SVD (Singular Value Decomposition). Retrieved from https://towardsdatascience.com/svd-8c2f72e264f

B., J. (2018, August 14). Clustering documents with TFIDF and KMeans. Retrieved from https://www.kaggle.com/jbencina/clustering-documents-with-tfidf-and-kmeans
Chen, E. (n.d.). Introduction to Latent Dirichlet Allocation. Retrieved from https://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/

Derksen, L. (2019, April 29). Visualising high-dimensional datasets using PCA and t-SNE in Python. Retrieved from https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

Galarnyk, M. (2020, May 1). PCA using Python (scikit-learn). Retrieved from https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

Li, S. (2018, June 1). Topic Modeling and Latent Dirichlet Allocation (LDA) in Python. Retrieved from https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

Maklin, C. (2019, July 21). TF IDF: TFIDF Python Example. Retrieved from https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76

Scott, W. (2019, May 21). TF-IDF for Document Ranking from scratch in python on real world dataset. Retrieved from https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089

Soma, J. (n.d.). Counting and stemming. Retrieved from http://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/counting-and-stemming/


---
layout: post
title:  "Machine Learning🤖: an introduction"
description: "An adventure in scikit-learn"
#categories: [python]
---


## What is Machine learning??

Machine Learning lies at the intersection of statistics, computer science and <strike>magic</strike> engineering 🔥.

With Machine learning (ML) we can gain insight from a dataset, to predict unseen future information. ML is turning data into information.

ML algorithms automate the decision making process without writing a specific set of rules for the decisions. It gives computers the ability to learn without being explicitly programmed. If computers don't learn, we have to program all the sets of rules. Learning, is more like farming: the farmer puts the seeds and some water, does the irrigation system and nature does most of the work to grow the crops.

The models derived in ML turn a small amount of input knowledge into a large amount of output knowledge. The goal of machine learning is to generalize beyond the examples in the training set. In order to do this, the learner must have knowledge or assumptions beyond the data it’s given in order to generalize.

ML problems can be divided in two categories: **supervised learning** and **unsupervised learning**. 

In **Supervised Learning**, we have a dataset consisting of both features and labels. The task is to construct an estimator which is able to predict the label of an object given the set of features. 

**Unsupervised Learning** addresses a different sort of problem. Here the data has **no labels**, and we are interested in finding similarities within the data. You can think of unsupervised learning as a means of finding labels from the data itself.

Supervised learning can be divided into two categories, **classification** and **regression**. 
In classification, labels are discrete, while in regression, labels are continuous (i.e. time series of stock prices). 

Unsupervised learning comprises tasks such as *dimensionality reduction*, *clustering*, and *density estimation*. All tasks that try to give meaning to unlabelled data.


## Scikit-learn

[Scikit-learn](http://scikit-learn.org) is a machine learning library that implements well designed machine learning algorithms. It provides interfaces between the different algorithms that allow us to switch between algorithms without too much effort (more of this in the examples).

Scikit-Learn is built upon Python's [NumPy (Numerical Python)](http://numpy.org) and [SciPy (Scientific Python)](http://scipy.org) libraries, which enable efficient in-core numerical and scientific computation within Python.

## Data Representation in Scikit-learn

Before we dig into the models it is a good idea to describe how data is represented inside this library. Most machine learning algorithms implemented in scikit-learn expect data to be stored in a
**two-dimensional array or matrix**. 

The size of the array is expected to be `[n_samples, n_features]`

- where X (**n_samples:**) is the number of samples,
- and y (**n_features:**) is the number of features that can be used to describe each process or item in analysis, 
  
  Usually we can create a model using an __estimator__ object and pass it both samples and features ``model.fit(X,y)``. 

## Internal data
Data in __scikit-learn__, as explained in the [documentation](http://scikit-learn.org/stable/tutorial/statistical_inference/index.html), is stored in ``from sklearn import datasets``. There are three options to access it:

- **Load Data:** some small datasets are packaged within the scikit-learn installation,
  and can be loaded using the command ``sklearn.datasets.load_*``
- **Fetch Data:** this command helps to download larger datasets It can be found in
  ``sklearn.datasets.fetch_*``
- **Make Data:** this command is able to generate data from models based on a
  random seed, using the command ``sklearn.datasets.make_*``

## Supervised learning examples with Scikit-learn

Now it is time for some hands on scikit-learn to really understand how we can make sense of some data, describing it with a model.

In the following examples 👇 we will analyse some toy datasets using both supervised and unsupervised learning methodologies.

The method `model.fit()` fits training data, calculating the internal parameters of the model, and is common to all the estimators. For supervised learning applications, this method accepts two arguments: the data `X` and the labels `y`. For unsupervised learning applications, it accepts only a single argument, the data `X`.

In the regression examples we are going to fit a line to some data ``(X,y)`` to predict with `model.predict()` the labels ``y_fit`` of a new set of data ``X_fit``.

The method `model.score()` shows a number between 0 and 1, indicating how good is the fit (1 is best). This method can be used in regression and classification problems. Here we will experiment a bit only with regression problems.
Remember that this is not the best way to evaluate a model: a very high score might indicate that the model is overfitting the data.

#### Here we create some data to use to fit our models


{% highlight python %}
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# randomly sample the data
np.random.seed(4)
X = 10 * np.random.random(size=(100,1))
y = -1 * X ** 2 + 100  + np.random.normal(1, 10, X.shape)
y = y.squeeze()
with plt.xkcd():
    plt.plot(X, y, 'o')
    plt.xticks([]), plt.yticks([])
{% endhighlight %}

![png](/assets/img/scikit-learn-intro/output_6_0.png)


#### Now we will fit a Linear Regression model.


{% highlight python %}
from sklearn.linear_model import LinearRegression
modelLR = LinearRegression()
modelLR.fit(X, y)
print(modelLR.score(X,y))

# Plot the data and the model prediction
X_fit = np.linspace(0, 10, 100)[:, np.newaxis]
y_fit = modelLR.predict(X_fit)

with plt.xkcd():
    plt.plot(X.squeeze(), y, 'o')
    plt.plot(X_fit.squeeze(), y_fit)
    plt.xticks([]), plt.yticks([])
{% endhighlight %}

    0.845422818275



![png](/assets/img/scikit-learn-intro/output_8_1.png)


#### Now we will fit a Random Forest model.


{% highlight python %}
    from sklearn.ensemble import RandomForestRegressor
    modelRFR = RandomForestRegressor()
    modelRFR.fit(X, y)
    print(modelRFR.score(X,y))
    # Plot the data and the model prediction
    X_fit = np.linspace(0, 10, 100)[:, np.newaxis]
    y_fit = modelRFR.predict(X_fit)

    with plt.xkcd():
        plt.plot(X, y, 'o')
        plt.plot(X_fit, y_fit)
        plt.xticks([]), plt.yticks([])
{% endhighlight %}

    0.979352874072



![png](/assets/img/scikit-learn-intro/output_10_1.png)


#### Now we will fit a Decision Tree model.


{% highlight python %}
    from sklearn.tree import DecisionTreeRegressor

    modelDTR = DecisionTreeRegressor(max_depth=5)
    modelDTR.fit(X, y)
    print(modelDTR.score(X,y))

    # Plot the data and the model prediction
    X_fit = np.linspace(0, 10, 100)[:, np.newaxis]
    y_fit = modelDTR.predict(X_fit)

    with plt.xkcd():
        plt.plot(X, y, 'o')
        plt.plot(X_fit, y_fit)
        plt.xticks([]), plt.yticks([])
{% endhighlight %}

    0.95832096062



![png](/assets/img/scikit-learn-intro/output_12_1.png)


#### Now we will fit a K-Nearest Neighbors model.


{% highlight python %}
    from sklearn.neighbors import KNeighborsRegressor


    modelKNN = KNeighborsRegressor(5)
    modelKNN.fit(X, y)
    print(modelKNN.score(X,y))

    # Plot the data and the model prediction
    X_fit = np.linspace(0, 10, 100)[:, np.newaxis]
    y_fit = modelKNN.predict(X_fit)

    with plt.xkcd():
        plt.plot(X, y, 'o')
        plt.plot(X_fit, y_fit);
        plt.xticks([]), plt.yticks([])
{% endhighlight %}
    0.927674602007



![png](/assets/img/scikit-learn-intro/output_14_1.png)


#### Now we will fit a Multi-layer Perceptron model.


{% highlight python %}
    from sklearn.neural_network import MLPRegressor

    modelMLP = MLPRegressor(solver='lbfgs', hidden_layer_sizes=100,
                            max_iter=250, shuffle=True, random_state=1)
    modelMLP.fit(X, y)
    print(modelMLP.score(X,y))

    # Plot the data and the model prediction
    X_fit = np.linspace(0, 10, 100)[:, np.newaxis]
    y_fit = modelMLP.predict(X_fit)

    with plt.xkcd():
        plt.plot(X, y, 'o')
        plt.plot(X_fit, y_fit);
        plt.xticks([]), plt.yticks([])
{% endhighlight %}
    0.920154566012



![png](/assets/img/scikit-learn-intro/output_16_1.png)


#### Now we will fit an Support Vector Regression model.


{% highlight python %}
    from sklearn.svm import SVR
    modelSVR = SVR(kernel='rbf', C=1e3, gamma=0.01)
    modelSVR.fit(X, y)
    print(modelSVR.score(X,y))

    # Plot the data and the model prediction
    X_fit = np.linspace(0, 10, 100)[:, np.newaxis]
    y_fit = modelSVR.predict(X_fit)

    with plt.xkcd():
        plt.plot(X, y, 'o')
        plt.plot(X_fit, y_fit)
        plt.xticks([]), plt.yticks([])
{% endhighlight %}
    0.915099262512



![png](/assets/img/scikit-learn-intro/output_18_1.png)


Thanks to the scikit-learn interface, it was really painless to test several different models. First we imported and created the model, then we fit the model to the data (X,y). Finally the new data X_fit has been used in the predict metod to predict the new labels y_fit.

In future posts we will use regression to predict some time series, and we will dig deeper into several models.

## Unsupervised Learning examples with Scikit-learn

In this section we are going to experiment with **unsupervised estimators**. Scikit learn has a different interface for this type of estimators.
After a the fit method has been called there are two options: `model.predict()` predicts labels in clustering algorithms and `model.transform()` will transform new data into a new basis. 
Some estimators implement a more efficient version of the method ``transform()`` called `fit_transform()`which more efficiently performs a fit and a transform on the same input data.

In the next sections we are going to look at dimensionality reduction and clustering techniques.

### Dimensionality reduction with PCA

Principal Component Analysis is an unsupervised method for *dimensionality reduction* in data.  Let's visualize how it works by looking at a two-dimensional case:


{% highlight python %}
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X, y = make_blobs(n_samples=400, centers=8,
                    random_state=6, cluster_std=0.9)
    with plt.xkcd():
        pca.fit(X)
        plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.5)
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)
            plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
            plt.axis('equal');
            plt.xticks([]), plt.yticks([])

{% endhighlight %}

![png](/assets/img/scikit-learn-intro/output_23_0.png)


The longer vector describes the "main direction", or trend, in the data. This tells us that the long vector axis is somehow more important.

The short vector or second principal component can be therefore ignored without much loss of information. PCA will "squeeze" the short vector on the long one using a projection. 


{% highlight python %}
    rv = PCA(0.65) # keep 65% of variance
    X_tr = rv.fit_transform(X)
    print(X.shape)
    print(X_trans.shape)

    with plt.xkcd():
        X_new = rv.inverse_transform(X_tr)
        plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.2)
        plt.plot(X_new[:, 0], X_new[:, 1], 'ob', alpha=0.8)
        plt.axis('equal');
        plt.xticks([]), plt.yticks([])
{% endhighlight %}
    (400, 2)
    (200, 1)



![png](/assets/img/scikit-learn-intro/output_25_1.png)


The transformed data is now projected all on the long axis reducing the dimension of the dataset to one. Later we will see how we can reduce the dimensions of a larger dataset with PCA and other methods.

Choosing the number of components depends also on the application. Intuitively the less number of components the more information we are loosing. Looking at the **explained variance** as a function of the components we can understand the information lost.


{% highlight python %}
    pca = PCA().fit(X)

    with plt.xkcd():
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
{% endhighlight %}

![png](/assets/img/scikit-learn-intro/output_28_0.png)


### Clustering with K-Means

K-Means is an algorithm for **unsupervised learning**. K-Means searches for cluster centers which are the mean of the points within them, such that every point is closest to the cluster center it is assigned to.

Let's look at how K-Means operates on a simple cluster of 6 groups.


{% highlight python %}
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=200, centers=6,
                    random_state=6, cluster_std=0.9)
    with plt.xkcd():
        plt.scatter(X[:, 0], X[:, 1], s=50);
        plt.xticks([]), plt.yticks([])
{% endhighlight %}

![png](/assets/img/scikit-learn-intro/output_31_0.png)


 In the figure above there are 6 clusters of points. The algorithm will try to label the data showing the **6 different clusters**. Without looking at the figure below, can you see the clusters?


{% highlight python %}
    from sklearn.cluster import KMeans
    modelKMNS = KMeans(6)  # 6 clusters
    modelKMNS.fit(X)
    y_kmeans = modelKMNS.predict(X)

    with plt.xkcd():
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow');
        plt.xticks([]), plt.yticks([])
{% endhighlight %}

![png](/assets/img/scikit-learn-intro/output_33_0.png)


Have you guessed the same clusters? 😃

### High dimensionality reduction

Now we will look at a higher dimensional example using the digits dataset included in scikit-learn. This dataset has 64 dimensions and we will try to reduce it to 2 so we can plot it. Dimensionality reduction is not only good for visualization, it also help in fitting better models.

{% highlight python %}
    from sklearn import datasets
    from matplotlib import offsetbox
    from sklearn import manifold

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    print(X.shape) # number of samples in number of dimensions
    print(y) # features

    # Scale and visualize the embedding vectors
    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                    color=plt.cm.Set1(y[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})


        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            
        plt.xticks([]), plt.yticks([])
        plt.title(title)


        (1797, 64)
        [0 1 2 ..., 8 9 8]
{% endhighlight %}

### PCA

Now that the dataset is loaded we can use PCA to reduce the dataset to 2 components.


{% highlight python %}
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    with plt.xkcd():
        plot_embedding(X_pca, "PCA embedding of the digits")
        plt.show()
{% endhighlight %}


![png](/assets/img/scikit-learn-intro/output_40_0.png)


### Isomap

[Isomap](http://scikit-learn.org/stable/modules/manifold.html#isomap) is a nonlinear manifold learning technique based on a sparse graph algorithm. Isomap seeks a lower-dimensional embedding which maintains geodesic distances between all points.


{% highlight python %}
iso = manifold.Isomap(n_components=2) #reduces 64 to 2 components

X_iso = iso.fit_transform(X)

with plt.xkcd():
    plot_embedding(X_iso, "Isomap embedding of the digits")
    plt.show()
{% endhighlight %}

![png](/assets/img/scikit-learn-intro/output_42_0.png)


###  t-SNE
t-distributed Stochastic Neighbor Embedding ([t-SNE](http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne)), converts affinities of data points to probabilities. The affinities in the original space are represented by Gaussian joint probabilities and the affinities in the embedded space are represented by Student’s t-distributions. While Isomap is best suited to unfold a single continuous low dimensional manifold, t-SNE will focus on the local structure of the data and will tend to extract clustered local groups of samples as highlighted on the S-curve example. This ability to group samples based on the local structure might be beneficial to visually disentangle a dataset that comprises several manifolds at once as is the case in the digits dataset. 



{% highlight python %}
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0) #reduces 64 to 2 components
X_tsne = tsne.fit_transform(X)

with plt.xkcd():
    plot_embedding(X_tsne, "t-SNE embedding of the digits")
    plt.show()
{% endhighlight %}

![png](/assets/img/scikit-learn-intro/output_44_0.png)


We can see how this algorithm groups even better the digits into separated clusters.

All the above techniques use the same interface ``fit_transform``, which is a clever combination of ``fit`` and ``transform``, that offers better performances. The method transform, given an unsupervised model (PCA, Isomap or t-SNE in this case), transforms new data into the new basis (from 64 to 2 dimensions). This method also accepts one argument `X`, and returns the new representation of the data based on the unsupervised model.

This gives us an idea of the relationship between the digits **without reference** to the labels.

In future posts we will look at how to evaluate models using cross validation and learn how to analyze the best ones👍 

We will also look at how to use unsupervised learning techniques to do feature engineering and boost model performance 🚀 

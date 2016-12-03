---
layout: post
title: Machine Learning nano-degree notebook (Unsupervised Learning)
excerpt: "Udacity machine learning nano-degree notebook."
categories: [Machine Learning]
tags: [Machine Learning, Deep Learning, Statistics, Data Science, Python]
comments: true
image:
  feature: ml-pic.jpg
---


# Unsupervised Learning

## 1. Clustering

[Sklearn Clustering Algorithm Interface](http://scikit-learn.org/stable/modules/clustering.html)

#### 1.1. Single Linkage Clustering
The simplest clustering which has following features:

  - consider each object a cluster (n objects).  
  - define inter-cluster distance as the distance between the closest two points in the two clusters. (Can be average or farthest two points, they have different name)  
  - merge two closest clusters  
  - repeat $$n-k$$ times to make k clusters  

This gives us a hierarchical agglomerative tree structure.

#### 1.2. Soft Clustering

Instead of finding out which data point belongs to which cluster like k-mean algorithm, in soft clustering, we will try to find out what the probability of a specific data point belongs to some hypothesis (which is the mean of some gaussian).

Assume the data was generate by:

  1. Select one of K gaussians (with fixed k means and variance) uniformly (So the prior can be ignored)
  2. Sample $$X_i$$ from that gaussian
  3. Repeat n times

Task: Find a hypothesis $$h = <\mu_1,...,\mu_k>$$ that maximize the probability of the data (Maximum likelihood)

P.S. Hidden variable is the variables that are inferred from other observed variables

#### 1.2 Expectation Maximization

**Expectation (define Z from $$\mu$$ and it is soft clustering, analogy of assigning data to the cluster in K-mean algorithm):**  

$$\mathbf{E}[Z_{i,j}] = \frac{P(x=x_i|\mu=\mu_j)}{\sum_{i=1}^{k} P(x=x_i|\mu=\mu_j)}$$

$$Z_{i,j}$$ stands for the probability of observed $$x_i$$ produced by the gaussian with $$\mu_j$$

Additionally:

$$P(x=x_i|\mu=\mu_j) = e^{-\frac{1}{2}\sigma^2(x_i-\mu_i)^2}$$

**Maximization (define $$\mu$$ from Z, analogy of computing mean each iteration in K-mean algorithm)**

$$\mu_{j} = \frac{\sum_i \mathbf{E}[Z_{i,j}]x_i}{\sum_{i} \mathbf{E}[Z_{i,j}]}$$



This can be transformed to K-mean algorithm if cluster assignments use argmax (which ends up with only 0 and 1, 0 stands for not belonging to that class and 1 otherwise).

**Properties of EM:**

- monotonically non-decreasing likelihood
- does not converge (practically does)
- will not diverge
- can get stuck (can randomly restart)
- works with any distribution (if E (Bayes net stuff), M (Counting things) solvable)

#### 1.3. Clustering Properties

- Richness  
  For any assignment of objects to clusters, there is some distance matrix D such that $$P_D$$ return that clustering $$\forall \space C \space \exists D P_D=C$$  
- Scale-invariance  
  Scaling distance by a positive value does not change the clustering $$\forall D \space \forall K>0 P_D = P_{KD}$$

- Consistency  
  Shrinking intra-cluster distances and expanding inter-cluster distances does not change the clustering $$P_D = P_{D^{'}}$$  

So what consistency says is if you found that a bunch of things were similar, and a bunch of other things were dissimilar, that if you made the things that were similar more similar, and the things that were not similar less similar, it shouldn't change your notion which things are similar and which things are not.

#### 1.4. Impossibility Theorem

No clustering schema can achieve all above three properties. These three properties are mutually contradiction in a sense. (Proven by Kleinberg)

#### Summary

- Clustering  
- Connection to compact description  
- Algorithm  
  - K-means  
  - SLC (terminates fast)  
  - EM (soft clusters)  
- Clustering properties & impossibility


*--------------------------------------------------------------------------------------- Updating... Nov. 30, 2016*  

## 2. Feature Engineering

#### 2.1. Min-Max Feature Scaling

Unbalanced features will cause problem, for example, height and weight of an person. The numeric meaning of height and weight is quite off and should never be operated by using plus or somethings. That's why we need feature scaling to make them somehow in a balanced space. (Ususally between 0 and 1)

Typically, the formula for feature scaling is like:

$$x^{'}=\frac{x-x_{min}}{x_{max}-x_{min}}$$

But outlier will mess up the rescaling if use this formula.

#### 2.2. Feature Selection

- Knowledge Discovery
  - Interpretability
  - Insight

- Curse of Dimensionality

Feature Selection can be exponentially hard since their are exponential number of feature subset for a given number of features.

Two potential algorithms that do feature selection: Filtering and Wrapping

##### 2.2.1. Filtering

Image feature searching is a black box, we input our features into this black box, and it will output the subset of features that this black box thinks are most important.


![]({{ site.url }}\assets\mlnd\filtering.png)

The search box can be any feature selection criteria. For example, a Decision Tree, the criteria will be information gain.

##### 2.2.2. Wrapping

In contrast with filtering, wrapping is trying to select a subset of features and train the model inside the box.

![]({{ site.url }}\assets\mlnd\wrapping.png)

It extremely time consuming. But still, there are a couple of way to avoid such huge time consumption:  
- Forward  
  - try a single feature among all features at first, evaluate and choose the best  
  - pick up more features and repeat until the evaluation result has no significant changes.  

- Backward  
  - try full set of features at first and evaluate.  
  - gradually reduce features one by one and evaluate. Stop when there is no significant changes.  

- Randomized Optimization  
  - try the randomized opt algorithms that are available.

#### 2.3. Relevance

- $$x_i$$ is strongly relevant if removing it degrades the Bayes optimal classifier (B.O.C).  
- $$x_i$$ is weakly relevant if:  
  - not strongly relevant  
  - $$\exists$$ subset of features S such that adding $$x_i$$ to S improves B.O.C.  
- $$x_i$$ is otherwise irrelevant  

Relevance is actually about information.  

#### 2.4. Relevance vs Usefulness

- Relevance measures effect on B.O.C.  
- Usefulness measures effect on a particular prediction.  

Usefulness is more about error instead of infomation/model/learner.  


#### Summary

1. Feature Selection  
2. Filtering (faster but ignore bias) vs Wrapping (slow but useful)  
3. Relevance (strong vs weak) vs Usefulness


*--------------------------------------------------------------------------------------- Updating... Dec. 2, 2016*  

## 3. Dimensionality Reduction

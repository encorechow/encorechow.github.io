---
layout: post
title: Machine Learning nano-degree notebook (Supervised Learning)
excerpt: "Udacity machine learning nano-degree notebook."
categories: [Machine Learning]
tags: [Machine Learning, Deep Learning, Statistics, Data Science, Python]
comments: true
image:
  feature: ml_pictogram.png
---


# Basics

## 1. Some Concepts

### Day 1

#### 1.1. All about mode, median and average

Given a set of data, suppose we have frequencies as Y axis and numbers as X axis. After we Draw out a histogram:
![Figure1]({{ site.url }}/assets/hist.png)

Here is Some concepts in terms of statistics:

- The value at which frequency is highest is called **Mode** (Mode can be a range that occurred with the highest frequency)
- Value in the middle is called **Median**
- Average (Mean)
Above three values can help describe the distributions of the data set.

##### 1.1.1. Mode

*Uniform Distributions* has no mode. Some of the distributions even have multiple modes. Mode depends on how you present your data.

##### 1.1.2. Mean
Always using the formula $$\bar{x} = \frac{\sum x}{n}$$ to calculate the mean of data set. Followings are the properties of mean:

- All scores in the distributions affect the mean.
- The mean can be described with a formula.
- Many samples from the same population will have similar mean.
- The mean of a sample can be used to make inferences about the population it came from.

##### 1.1.3. Median

the Median of a data set is robust against outlier.

##### Summary

Mean, Median, Mode describe the center of the distributions, so they are measures of center. Sometimes mean doesn't describe the center because of outlier, and Mode doesn't describe the center because mode depends on how you present your data, and Median doesn't describe the center because it doesn't take every data point into account. Sometimes in order to avoid the influence of outlier, data scientist usually cut off 25% upper tails of the data and 25% lower tails.


#### 1.2. Range and outlier

##### 1.2.1. Range

Inter-quantile range: First split the data into two halves, then calculate the median of the upper half and lower half respectively. Let's say the upper median is Q1, the lower median is Q3, and the median of data set is Q2. Inter-quantile is the range between Q1 to Q3 (IQR). IQR has the properties that
1. About 50% of the data falls within it;
2. The IQR is not affected by outliers.

##### 1.2.2. Outlier

A datum x can be consider as an outlier when (Q1 is the first quantile and Q3 is the third quantile):

1. For outlier x, $$x < Q1 - 1.5\times(IQR)$$;
2. For outlier x, $$x > Q3 + 1.5\times(IQR)$$;


#### 1.3. Distributions

Normal Distributions ($$\sigma$$ is standard deviation):

- 68% percent of data fall between $$\bar{x}-\sigma$$ and $$\bar{x}+\sigma$$.
- 95% percent of data fall between $$\bar{x}-2\times\sigma$$ and $$\bar{x}+2\times\sigma$$.

Bessel's Correction:

Divided by N-1, we get a bigger number of standard deviation. Since the samples tend to be values in the middle of the population. In this case , the variability (variance) of the samples will be less than the variability of the entire population. Therefore we correct the standard deviation to make it a little bit bigger by put N-1 in the denominator: $$SD = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \overline{x})^2}$$, which is called **Sample Standard Deviation**.

---

### Day 2

#### 1.4. Data Type

- Numeric Data
    - Data have exactly numbers as measurement.
    - Discrete or continuous.

- Categorical Data
    - Represent characteristics.
    - Can take numeric value, but don't have mathematical meaning.
    - Ordinal data

- Time-series data
    - Data collected via repeated measurements over time. (date, timestamp)

- Text
    - Words


#### 1.5. Bias and Variance

- High Bias means the model has high error on training data set (low &&R^2&&, large SSE), the model pays little attention to data, it's oversimplified.
- High variance means the model has much higher error on testing data set but low error on training data set, since it pays too much attention to data (Does not generalize well), which cause overfit.


#### 1.6. Curse of Dimensionality

As the number of features or dimensionality grows, the amount of data that we need to generalize accurately also grows exponentially.


---

# Supervised Learning

Training a model on a set of data points with their actual labels or values.

## 2. Regression & Classification

### Day 3

#### 2.1. Polynomial Regression

Suppose X (only one feature) is a matrix of data points and W is a weight vector, and Y is the vector of target value for corresponding value in X:

$$X = \begin{bmatrix}
    x_{1} & x_{1}^2 & x_{1}^3 & \dots  & x_{1}^n \\
    x_{2} & x_{2}^2 & x_{2}^3 & \dots  & x_{2}^n \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{d} & x_{d}^2 & x_{d}^3 & \dots  & x_{d}^n
\end{bmatrix}$$,
$$W = \begin{bmatrix}
    w_{1} \\
    w_{2} \\
    \vdots \\
    w_{d}
\end{bmatrix}$$

We can solve $$ X\times W = Y$$ to find what exactly W is by following calculus:

$$ XW \approx Y \\
X^TXW \approx X^TY \space \text{(X^TX will have an inverse)}\\
(X^TX)^{-1}X^TXW \approx (X^TX)^{-1}X^TY \\
W \approx (X^TX)^{-1}X^TY
$$
That gives us the exact coefficient that we need to do the polynomial regression.


#### 2.2. IID

There is a huge fundamental assumption for a lot of algorithms which is so called IID. IID stands for independent identical distribution, which assumes that all training, testing and real world data for a problem comes from independent and identical distribution. Otherwise we can not fit a model well.

---

### Day 4

#### 2.3. Linear Regression

The best regression is the one that:

$$\DeclareMathOperator*{\minimize}{minimize}
\displaystyle{\minimize \sum_{\text{all training points}} (\text{actual} - \text{predicted})^2}
$$

Several algorithms to minimize sum of squared errors:
- ordinary least squares (OLS) (used by sklearn)
- Gradient decent

Why squared errors? There can be multiple lines that minimize $$\sum \|error\|$$, but only one line will minimize $$\sum error^2$$. Furthermore, Using SSE also makes implementation much easier. But one problem of SSE is that larger SSE doesn't mean the model doesn't fit well because the SSE is guarantee to grow as we add more data. Doing comparison on two models over different number of data will be ambiguous.  

##### 2.3.1. $$R^2$$ "R squared" of a regression

$$R^2$$ answers the question "how much of my change in the output (y) is explained by the change in my input(x)".
$$0.0 < R^2 < 1.0$$
If R^2 is small, that means the model doesn't fit very well since the model doesn't capture trend in data. If R^2 is large, that means the model does a good job of describing relationship between input(x) and output(y).

Here is a comparison matrix of Classification and Regression:

| Properties                  | Supervised Classification | Regression                        |
|:--------------------------- |:--------------------------|:----------------------------------|
| output type                 | discrete (class labels)   | continuous (number)               |
| what are you tring to find  | decision boundary         | "bset fit line (hyperplate)"      |
| evaluationo                 | accuracy                  | "sum of squared error" or $$R^2$$ |


#### 2.4. Parametric Regression ,Multi-variate Regression, k nearest neighbor(KNN), Kernel Regression

- Parametric Regression has the form like: $$y = m_2\times X^2 + m_1\times X + b$$, It considers the parameters of the model and after training process, it tosses training data away and just use the parameters later on to predict the queries.
- Multi-variate Regression has the form like: $$y = W_1X_1 + W_2X_2 + ... + W_nX_n$$, which means you have multiple variable in the model. It belongs to parametric regression.

- k nearest neighbor(KNN) finds K nearest points and calculate the mean y value to get the result. In KNN, each neighbor normally has same weight.
- Kernel Regression differ from KNN in the way that kernel regression weighted each nearest neighbor point to take their importance into account.

KNN and Kernel Regression are both instance based method (non-parametric) where we keep the data and we consult when we make a query.

Parametric approach usually doesn't have to store original data. So it's space efficient. But we can't update the model when there are more upcoming data. Usually we have to do a complete re-run of training process. Therefore parametric approach is training slow but querying fast.
Non-parametric has to store all data points. It's hard to apply when we have a huge amount of data points. But new evidence (data point) can be added in very easily. Since no parameter needs to be learn.

---

## 3. Decision Tree

### Day 5
Decision tree is NP problem if we are trying to find all possible output of a problem like XOR expression.

#### 3.1. ID3 algorithms:

Loop:
  - A <- best attribute
  - Assign A as decision attribute for node
  - For each value of A, create a descendent of node
  - Sort training examples to leaves based upon exactly what the value takes on
  - If examples perfectly classified, Stop
  - Else iterate over leaves

What is best attribute? best attribute has maximum information gain.

$$ GAIN(S, A) = Entropy(S) - \sum_v \frac{\|S_v\|}{\|S\|}Entropy(S_v)$$

More mathematically:

$$I(Y, X) = H(Y) - H(Y|X)$$

$$H(Y|X) = \sum_x P(X = x)H(Y|X = x)$$


#### 3.2. ID3 bias

Restriction bias : All the hypothesis in hypothesis set H we will consider
Preference bias : What sort of hypothesis from the hypothesis set we prefer $$h \subset H$$

Inductive Bias: Given a whole bounch of decision trees, which dicision trees will ID3 prefer over others.
  - Good splits at top
  - Prefer correct over incorrect (model)
  - Shorter trees

#### 3.3. Continuous Attribute

e.g. Age, weight, distance. In decision tree, we can do range for each node if the attributes are continuous.

#### 3.4. When the decision tree stop?

- We can do pruning.

#### 3.5. Regression in decision tree

Actually either way we can do vote on the leaves.

---

### Day 6

Sklearn Decision Tree Interface: [Decision Tree](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

#### 3.6. Entropy

Entropy is very powerful thing, which controls how a decision tree decide where to split the data.

  - Definition: measure of impurity in a bunch of samples.

Mathematical formula of entropy: $$\sum_i -p_ilog_2(p_i)$$ (i is the totally number of class (for classification) in this node)

Again, Information gain is the opposite of entropy, which measure how much information we get if we split the data in a certain way.

  - Information Gain = entropy(parent) - [weighted average] * entropy(children)

Decision Tree algorithm will maximize information gain. By default, Decision Tree in Sklearn uses gini to measure the impurity of a node, which is slightly different from entropy and information gain.



## 4. Neural Network

#### 4.1. Perceptron

Suppose we have an input vector X and initial weight vector W; in addition we have a firing threshold $$\theta$$. We measure whether $$ activation(\sum_i^k X_iW_i)$$ is greater or smaller than $$\theta$$ to output the result. This is called perceptron.

##### Perceptron Training

Given examples, find weights that map input to outputs. We have two algorithms to solve this:

  - Perceptron rule (threshold)
  - Gradient descent / delta rule (un-threshold)

##### Perceptron Rule

Update rule:

y: target; $$\hat y$$: output; $$\mu$$: learning rate; x: input

$$W_i = W_i + \Delta W_i$$

$$\Delta W_i = \mu (y-\hat y)X_i$$

$$\hat y = (\sum_i w_ix_i \geq 0)$$


### Day 7

_little remainder:_

1. _The purpose of cross-validation is model checking, not model building. For example, let's say we have two model: liear regression and neural network. For a specific data set, if we want to know which one performs better than another, we should use cross-validation to do model checking and pick up the one that has better accuracy over the cross-validation data set._

2. _cross-validation is also helpful to find the better hyper-parameters. For example, Given a list of $$\lambda$$(hyper-parameter of regularization), we want to find out which value performs better among all for a trained model to reduce overfitting. In this scenario, cross-validation will do the trick._

#### 4.2. Gradient Descent

The basic principle of Gradient Descent is pretty similar to perceptron update rule. But instead of considering the output to be discrete, gradient descent take advantage of the continuous output and differentiate the loss function based on this continuous output with its actual label. Say we have a sum of squared loss $$L(w) = \frac{1}{2}\sum_{(x,y)\subset D} (y - a)^2$$. The purpose is to minimize the loss as much as we can. There are actually multiple ways to do that. Gradient Descent is one of them.

So in order to minimize the loss, we have to know the direction of current loss. By differentiating the loss function we can know exactly how to update our weights to reach the extreme point. For example, if the derivative of the loss function is greater than 0, we should decrease our weight in some amount for reaching the optima. Otherwise we increase our weight.

Learning rate controls how much we should update on our weights.

It has chance that the optimization of neural network ends up with local optima since the non-linearity of neural network and randomness of the initial weights.  

#### Summary

- Perceptron -> threshold
- Network can produce any boolean function
- Perceptron rule -> finite time for linearly separable data set
- General differentiable rule -> basic propagation and gradient descent
- Preference / restriction bias of neural network


## 5. Support Vector Machine

### Day 8

#### 5.1 SVM overview

In some situations, we try to not only find the line or hyper-plate that separates multiple classes of data, but we also seek for the best line that splits the data apart. The chart below shows three lines that achieve the purpose to separate a random data set.

{% include graphs/Machine-Learning-Nano-Degree-sample1.html %}  


Apparently, our intuition tells us that the green line separate the data set perfectly. We can observe that there are some points lie on the line1 and line2, which means if we use these two line as our boundaries, some of the points can be ambiguous when we classify them.

But what is the best boundary? As we can see in the graph, the distance between two blue lines tells us how far the most ambiguous points in ecah classes away from each other. This distance might be able to support us to figure out the best boundary.

Suppose we have class -1 and class 1 ($$y \subset \{-1, 1\}$$). If the points lie on the boundary (hyper-plate), our classify is not sure which class that point belongs to, so we will have:

$$ W^TX + b = 0 $$ (the green line)

and we can also say that:

$$W^TX + b = 1$$

$$W^TX + b = -1$$

For other two blue lines. We actually want to maximize the the distance between the two blue lines. Given a point $$x_1$$ lies on the line1 and point $$x_2$$ lies on the line2, after putting them into the equation and subtracting them we have:

$$W^T(x_1 - x_2) = 2$$

Now we want to know what $$x_1 - x_2$$ is (the distance between these two points). Intuitively we can divide left and right by W. Since W is a vector, There is not way to divide it at least in the really world. Instead of dividing W directly, we can divide the norm of W from both side. That gives us normalized version of W. So now we have:

$$\frac{W^T}{\|W\|}(x_1 - x_2) = \frac{2}{\|W\|}$$

Actually the whole thing on the left represents the distance of $$x_1$$ and $$x_2$$ projected on W direction. This is called margin. We want to find a line that maximizes the margin specifically, which is also $$\frac{2}{\|W\|}$$.

Then the problem becomes: maximize $$\frac{2}{\|W\|}$$ while classifying everything correctly. This problem also can be turned to minimize $$\frac{1}{2}\|W^2\|$$. The minimization problem will be easier since it's a quadratic problem. Quadratic programming problem has a very particular form:

$$W(\alpha) = \sum_i \alpha - \frac{1}{2}\sum_{ij}\alpha_i\alpha_jy_iy_jx_i^Tx_j$$

$$s.t. \alpha \geq \emptyset, \sum_i \alpha_iy_i = \emptyset$$

The points that contribute to the quadratic problem are support vector. That is where SVM comes from.

#### 5.2 Kernel

Sometimes the data set is not linearly separable. SVM solves this by projecting the data set to higher dimensional space. The way to do that is called kernel function. For example, quadratic kernel function is $$K = (X^TY)^2$$. There are a lot of kernel that in polynomial order, which looks like $$K = (X^TY + c)^p$$



#### Summary

- Margins ~ generalization & overfitting
- Maximize Margins (big is better)
- Optimization problem for finding max margins: quadratic problem.
- Support Vector.
- Kernel Trick (K(x, y) -> domain knowledge)



## 6. Non-parametric Models

### Day 9

#### 6.1. Instance Based Learning

In terms of instance based learning, there is actually no mathematical formula for the model. We predict the new instance by looking up the existed instances based on some criteria (e.g. Similarity)

##### 6.1.1. K nearest neighbor

Given:

- Training Data $$D = \{x_i, y_i\}$$
- Distance Metric $$d(q, x)$$
- Number of Neighbors K
- Query Point q

Compute:

- K smallest points for $$NN = \{i: d(q, x_i)\}$$

Return:

- Classification: majority vote (or weighted vote) s.t. $$y_i \subset NN$$ for q (plurality)
- Regression: mean (or weighted mean) s.t. $$y_i \subset NN$$ for q

#### 6.2 Comparison Matrix of Non-parametric Model and Parametric Model

<table>
  <tr>
    <th></th>
    <th>Time</th>
    <th>Space</th>
  </tr>
  <tr>
    <td rowspan="2">1-NN</td>
    <td>(Learning) 1</td>
    <td>(Learning) N</td>
  </tr>
  <tr>
    <td>(Query) logN (Binary Search)</td>
    <td>(Query) 1</td>
  </tr>
  <tr>
    <td rowspan="2">K-NN</td>
    <td>(Learning) 1</td>
    <td>(Learning) N</td>
  </tr>
  <tr>
    <td>(Query) logN + K</td>
    <td>(Query) 1</td>
  </tr>
  <tr>
    <td rowspan="2">Linear Regression</td>
    <td>(Learning) N</td>
    <td>(Learning) 1</td>
  </tr>
  <tr>
    <td>(Query) 1</td>
    <td>(Query) 1</td>
  </tr>
</table>

Basically, Instance based learning is called lazy learner, while parametric learning is called eager learner.


#### 6.3 Preference Bias for KNN

Remember preference bias is a notion of why we prefer one hypothesis over another, which is also our belief about what makes a good hypothesis.

- Locality -> near points are similar.
- Smoothness -> averaging.
- All features matters equally.


#### Summary

In KNN, Distance metric really matters a lot. We should always try to choose a suitable distance metric for a specific problem.

There is some other stuff that is worth mentioning. After we picked up a bunch of nearest neighbor, we could do a regression on these neighbors, which is called locally weighted regression.


## 7. Bayesian Methods

#### 7.1 Bayes Rule

Given some data and domain knowledge, we want to learn the best hypothesis for these data. That also means we want to learn the most possible hypothesis. That is, we are trying to find a hypothesis that:

$$ argmax_{h\subset H} Pr(h|D)$$

Bayes Rule can be expressed as following given observed data D to find the probability of h (hypothesis) based on D:

$$ Pr(h|D) = \frac{Pr(D|h)Pr(h)}{Pr(D)}$$

joint probability:

$$ Pr(a, b) = Pr(a|b)Pr(b) $$

$$ Pr(a, b) = Pr(b|a)Pr(a) $$

Actually the $$ Pr(D) $$ is the prior belief of seeing some particular set of data (prior on the data).

So $$ Pr(D|h) $$ can be explained as given a hypothesis h, what is the probability of the data.
The data is composed as $$ \{(X_i, label_i)\} $$. what that really means is, Given a set of $$ X_i $$, Given a real world that h is true, what is the likelihood that we will see the particular $$ label_i $$. This is like running hypothesis, for example, labeling the data manually.

$$ Pr(h) $$ is the prior on a particular hypothesis drawn from the hypothesis space. Which encapsulates our prior belief that one hypothesis is likely or unlikely compared with another hypothesis.

### Day 10

#### 7.2 Bayesian Learning


The algorithm of bayesian learning:

For each $$ h\subset H $$:  
  calculate: $$Pr(h|D) = \frac{Pr(D|h)Pr(h)}{Pr(D)}$$  
  Output: $$ h = argmax_h Pr(h|D)$$  

$$Pr(D)$$ can be ignored in terms of argmax.  
This is called MAP, namely, maximum a posterior.  

If we ingore $$Pr(h)$$, we are actually computing the maximum likelihood, which is:

$$ h_{ML} = argmax_h Pr(D|h) $$  

We are not actually dropping the $$Pr(h)$$. What we are really saying is the prior is uniform (1 / size of hypothesis), they are equally likely. For example, we have a classification problem that only has two classes and their prior is all 0.5. So they are equally likely. Thus $$Pr(h)$$ will not affect the result of argmax.



## 8. Ensemble Learning Boosting

### Day 11

#### 8.1. Ensemble Learning Simple Rules

The reason that we want to use ensemble is sometimes a single feature or a couple of features can not fully represent what a truth is going to be. Neural network can be treat as a part of ensemble model in some senses.

Ensemble learning consists of:

1. Learning over a subset of data  
  -> Uniformly randomly pick data, and apply a learner (Bagging)  
  -> Focusing on hard examples (Boosting)  
1. Combine the results together  
  -> mean, averaging (Bagging)  
  -> weighted mean (Boosting)



#### 8.2. Boosting  

Pseudocode of Boosting (Binary Classification):  

- Given a training set $$\{(x_i, y)\}, y \in \{1,-1\}$$
- For t = 1 to T:  
  - construct $$D_t$$  
  - find weak classifier $$h_t(x)$$ with small error $$\epsilon_t = Pr_{D_t}[h_t(x)=y_i]$$
- output $$H_{final}$$


##### 8.2.1. Construct Distribution

The most important part 1 of boosting is construct distribution:

- set up the distribution of first iteration to be uniform distribution, which is $$D_1(i) = \frac{1}{n}$$. Since we know nothing at beginning about the data set.  
- update the distribution by increasing the weight of the examples that are predicted wrong and decreasing otherwise:  

  $$D_{t+1}(i) = \frac{D_t(i)\times e^{-\alpha_ty_ih_t(x_i)}}{Z_t} \space where \space \alpha_t = \frac{1}{2}ln\frac{1-\epsilon_t}{\epsilon_t}$$

##### 8.2.2. Output Final Result

The final hypothesis will be the weighted hypothesis over all trained classifiers:

$$H_{final}(x) = sign(\sum_t \alpha_th_t(x))$$

##### 8.2.3. Minimize Overfitting

AdaBoost can minimize overfitting by adding up more and more weak learner. This ends up with a larger margin between the classified data, which means the model has more confidence to classify one point is one of the class over the other.

Some other things: pink noise (uniform distribution) will cause boosting overfit. (white noise is gaussian noise)

#### Summary

- ensembles are good.  
- bagging is good.
- combining simple -> complex.
- boosting is really good. -> agnostic to learner
- weak learner.
- error with distribution.

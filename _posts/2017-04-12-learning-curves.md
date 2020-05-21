---
layout: post
title:  "Learning curves"
description: "Bias/Variance Trade Off"
---

Learning curves answer the question: How does the accuracy of a learning method change as a function of
the training-set size?

It is a graph that compares the performance of a model on training and testing data over a varying number of training instances. It helps visually to identify if the model is underfitting (High Bias) or overfitting (High Variance).

Separating training and test sets and graphing them individually gives us an idea of how well the model can generalize to new data. We should generally see performance improve as the number of training points increases.

The key is to find the sweet spot that minimizes bias and variance by finding the right level of model complexity.

In this section, we'll work with a simple 1 dimensional regression problem. First we will analyse how different polynomial models perform and then, with the help of the learning curves, we will choose the model that performs the best.

In this section we will use, for simplicity, `numpy.polyfit` and `numpy.polyval`, and implement the learning curve algorithm. In the next section we will use scikit-learn with `sklearn.linear_model`,



{% highlight python %}
import numpy as np
%matplotlib inline

X = 10 * np.random.random(20)
y = 0.5 * X ** 2 - X + 1

p = np.polyfit(X, y, deg=2)
print('Model parameters:',p)
{% endhighlight %}

    Model parameters: [ 0.5 -1.   1. ]



{% highlight python %}
X_fit = np.linspace(-1, 12, 1000)
y_fit = np.polyval(p, X_fit)

plt.xkcd()
plt.scatter(X, y)
plt.plot(X_fit, y_fit)
{% endhighlight %}



![png](/assets/img/learning-curves/output_2_1.png)


Now, we will add Gaussian noise to the data to simulate a more realistic setting. We'll also calculate the RMS error of each resulting model on the input data.


{% highlight python %}
# Adding Gaussian noise
np.random.seed(42)
X = 10 * np.random.random(20)
y = 0.5 * X ** 2 - X + 1  + np.random.normal(0, 2, X.shape)

# ---> Fitting a line: deg 1
p = np.polyfit(X, y, deg=1)
X_fit = np.linspace(0, 10, 100)
y_fit = np.polyval(p, X_fit)

plt.scatter(X, y)
plt.plot(X_fit, y_fit)
plt.ylim(-10, 50)
print("RMS error deg1= %.4g" % np.sqrt(np.mean((y - np.polyval(p, X)) ** 2)))


# ---> Fitting a second order poly: deg 2
p = np.polyfit(X, y, deg=2)
X_fit = np.linspace(0, 10, 100)
y_fit = np.polyval(p, X_fit)

plt.plot(X_fit, y_fit)
plt.ylim(-10, 50)
print("RMS error deg2= %.4g" % np.sqrt(np.mean((y - np.polyval(p, X)) ** 2)))

# ---> Fitting a fith order poly: deg 5
p = np.polyfit(X, y, deg=5)
X_fit = np.linspace(0, 10, 100)
y_fit = np.polyval(p, X_fit)

plt.plot(X_fit, y_fit)
plt.ylim(-10, 50)
print("RMS error deg5= %.4g" % np.sqrt(np.mean((y - np.polyval(p, X)) ** 2)))

# ---> Fitting a 13th order poly: deg 13
p = np.polyfit(X, y, deg=13)
X_fit = np.linspace(0, 10, 100)
y_fit = np.polyval(p, X_fit)

plt.plot(X_fit, y_fit)
plt.ylim(-10, 50)
print("RMS error deg13= %.4g" % np.sqrt(np.mean((y - np.polyval(p, X)) ** 2)))

{% endhighlight %}

    RMS error deg1= 4.357
    RMS error deg2= 1.726
    RMS error deg5= 1.444
    RMS error deg13= 1.186



![png](/assets/img/learning-curves/output_4_1.png)


Increasing the degree of the polynomial, lowers the RMS error but does it guarantee a more robust model? What if we are overfitting?

One way to address this issue is to use what are often called **Learning Curves**.  We'd like to tune our value of the *hyperparameter* `d` (polynomial degree) to give us the best fit.

In the figure above 👆🏼, we see 4 different models. For `deg = 1`, the data is under-fit. This means that the model is too simplistic: no straight line will ever be a good fit to this data. In this case, we say that the model suffers from **high bias**. 

On the other hand the model with `deg = 13` the data is over-fit. In this case, we say that the model suffers from **high variance**. This means that the model has too many free parameters (13 in this case) which can be adjusted to perfectly fit the training data. If we add a new point to this plot, though, chances are it will be very far from
the curve representing the degree-6 fit.

With the learning curves we have a way to quantitatively identify bias and variance, and optimize the
metaparameters (in this case, the polynomial degree ``deg``) in order to determine the best algorithm. 

Now we will split the data in half, using some for training and some for validation.


{% highlight python %}
from sklearn.cross_validation import train_test_split

N = 200
f_crossval = 0.5

# randomly sample the data
np.random.seed(4)
X = 10 * np.random.random(N)
y = 0.5 * X ** 2 - X + 1  + np.random.normal(0, 2, X.shape)

# split into training and validation sets.
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=f_crossval)

# show the training and cross-validation sets
plt.scatter(xtrain, ytrain, color='red')
plt.scatter(xtest, ytest, color='blue')
legend( ('training', 'validation'), loc='upper left')

{% endhighlight %}



![png](/assets/img/learning-curves/output_6_1.png)


In blue🔵 we have the validation set, in red🔴 the training set.

## Learning Curves in Practice

A learning curve is a plot of the training and cross-validation
error as a function of the number of training points. Note that
when we train on a small subset of the training data, the training
error is computed using this subset, not the full training set.
These plots can give a quantitative view into how beneficial it
will be to add training samples.

The pseudo algorithm works as follows:

- Given training/test set partition
- for each sample size s on learning curve
    - (optionally) repeat n times
        - randomly select s instances from training set
        - learn model
        - evaluate model on test set to determine accuracy a or error e
        - plot (s, a/e) or (s, avg. accuracy and error bars) 


{% highlight python %}
def compute_error(x, y, p):
    yfit = np.polyval(p, x)
    return np.sqrt(np.mean((y - yfit) ** 2)) #r mean square

def plot_learning_curve(d):
    sizes = np.linspace(2, N, 50).astype(int)
    train_err = np.zeros(sizes.shape)
    crossval_err = np.zeros(sizes.shape)

    for i, size in enumerate(sizes):
        # Train on only the first `size` points
        p = np.polyfit(xtrain[:size], ytrain[:size], d)
        
        # Validation error is on the *entire* validation set
        crossval_err[i] = compute_error(xtest, ytest, p)
        
        # Training error is on only the points used for training
        train_err[i] = compute_error(xtrain[:size], ytrain[:size], p)

    fig, ax = plt.subplots()
    ax.plot(sizes, crossval_err, lw=2, color='blue', label='validation error')
    ax.plot(sizes, train_err, lw=2, color='red', label='training error')
    ax.plot([0, N], [error, error], '--k', label='intrinsic error')

    ax.set_xlabel('traning set size')
    ax.set_ylabel('rms error')
    legend( ('validation', 'training'), loc='lower right')
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 5)

    ax.set_title('d = %i' % d)
{% endhighlight %}

Now we can plot the learning curves for our 4 cases (d=1,d=2,d=5,d=13)


{% highlight python %}
plot_learning_curve(d=1)
plot_learning_curve(d=2)
plot_learning_curve(d=5)
plot_learning_curve(d=13)
{% endhighlight %}



![png](/assets/img/learning-curves/output_11_1.png)



![png](/assets/img/learning-curves/output_11_2.png)



![png](/assets/img/learning-curves/output_11_3.png)



![png](/assets/img/learning-curves/output_11_4.png)


### Learning curves d=1
First, let’s look at the performance on the training set. When there are just one or two instances in the training set, the model can fit them perfectly, which is why the curve starts at zero. But as new data is added to the training set, it becomes impossible for the model to fit the training data perfectly. That is why the training error flattens out.

On the other hand, in the validation curve, when the model is trained on very few training instances, it is incapable of generalizing properly, which is why the validation error is initially quite big. Then as the model is shown more
training examples, it learns and it will perform better at generalizing. Thus the validation error slowly goes down.

These learning curves are typical of a **high bias** underfitting model: the curves reached a point where the error is high and they are close to each other.

If your model is underfitting the training data, adding more training examples will not help. You need to use a more complex model or come up with better features.

### Learning curves d=13
In this figure the error on the training data is much lower than the high bias case, and there is a big gap between the curves. This means that the model performs significantly better on the training data than on the validation data.

A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have **high variance**, and thus to overfit the training data. As we add more samples to this training set, the training error will continue to increase, while the cross-validation error will continue to decrease, until they meet in the middle. 


### Learning curves d=2
We already knew from the beginning that a quadratic model will be the best fit for our fake dataset. You can see that both curves are close to each other and the error is fairly low.

Increasing a model’s complexity will typically increase its variance and reduce its bias. Conversely, reducing model complexity increases its bias and reduces its variance. This is why it is called a tradeoff.



### Final notes

If your algorithm shows **high bias**:

- **Add more features**. Adding these features to the training and test sets can improve
  a high-biased model,
- **Use a more sophisticated model**. Adding complexity to the model can
  help improve on bias.
- **Decrease regularization**. If a model has high bias,
  decreasing the effect of regularization can lead to better results, because you would have a smaller regularized term, giving more importance to other features.
  
If your algorithm shows **high variance**:

- **Use fewer features**. Using a feature selection technique may be
  useful, and decrease the model over-fitting,
- **Use more training samples**. Adding more data can reduce
  the effect of over-fitting, and lead to improvements in a high
  variance model,
- **Increase Regularization**. In a high-variance model, increasing regularization
  can lead to better results, because you would have a larger regularized term, giving less importance to other features.


Remember thath in the same way that parameters can be over-fit to the training set, also ``d`` can
be over-fit to the validation set.  Because of this, the validation error tends to under-predict the classification error of new data.

For this reason, it is recommended to split the data into three sets:

- The **training set**, used to train the model (usually ~60% of the data)
- The **validation set**, used to validate the model (usually ~20% of the data)
- The **test set**, used to evaluate the expected error of the validated model (usually ~20% of the data)

This may seem excessive, and many machine learning practitioners ignore the need for a test set.  But if your goal is to predict the error of a model on unknown data, using a test set is vital, and it will lead to a more robust model.



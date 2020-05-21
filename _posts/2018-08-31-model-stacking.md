---
layout: post
title:  "Model stacking"
description: "Some personal notes"
---

## What is stacking?

Stacking (also called meta ensembling) is a model ensembling technique used to combine information from multiple predictive models to generate a new model. Often times the stacked model (also called 2nd-level model) will outperform each of the individual models due its smoothing nature and ability to highlight each base model where it performs best and discredit each base model where it performs poorly. For this reason, stacking is most effective when the base models are significantly different. Here I provide a simple example and guide on how stacking is most often implemented in practice.

The main point to take home is that we’re using the predictions of the base models as features (i.e. meta features) for the stacked model. So, the stacked model is able to discern where each model performs well and where each model performs poorly. It’s also important to note that the meta features in row i of train_meta are not dependent on the target value in row i because they were produced using information that excluded the target_i in the base models’ fitting procedure.

Alternatively, we could make predictions on the test dataset using each base model immediately after it gets fit to each test fold. In our case this would generate test-set predictions for five K-Nearest Neighbors models and five SVM models. Then we would average the predictions per model to generate our M1 and M2 meta features. One benefit to this is that it’s less time consuming than the first approach (since we don’t have to retrain each model on the full training dataset). It also helps that our train meta features and test meta features should follow a similar distribution. However, the test metas M1 and M2 are likely more accurate in the first approach since each base model was trained on the full training dataset (as opposed to 80% of the training dataset, five times in the 2nd approach).


## Stacked Model Hyper Parameter Tuning
So, how do you tune the hyper parameters of the stacked model? Regarding the base models, we can tune their hyper parameters using Cross-Validation + Grid Search just like we did earlier. It doesn’t really matter what folds we use, but it’s usually convenient to use the same folds that we use for stacking. Tuning the hyper parameters of the stacked model is where things get interesting. In practice most people (including myself) simply use Cross Validation + Grid Search using the same exact CV folds used to generate the Meta Features. There’s a subtle flaw to this approach – can you spot it?

Indeed, there’s a small bit of data leakage in our stacking CV procedure. Consider the 1st round of Cross Validation for the stacked model. We fit a model S to {fold2, fold3, fold4, fold5}, make predictions on fold1 and evaluate performance. But the meta features in {fold2, fold3, fold4, fold5} are dependent on the target values in fold1. So, the target values we’re trying to predict are themselves embedded into the features we’re using to fit our model. This is leakage and in theory S could deduce information about the target values from the meta features in a way that would cause it to overfit the training data and not generalize well to out-of-bag samples. However, you have to work hard to conjure up an example where this leakage is significant enough to cause the stacked model to overfit. In practice, everyone ignores this theoretical hole (and frankly I think most people are unaware it even exists!).

## Stacking Model Selection and Features
How do you know what model to choose as the stacker and what features to include with the meta features? In my opinion, this is more of an art than a science. Your best bet is to try different things and familiarize yourself with what works and what doesn’t. Another question is, what (if any) other features should you include in for the stacking model in addition to the meta features? Again this is somewhat of an art. Looking at our example, it’s pretty evident that DistFromCenter plays a part in determining which model will perform well. The KNN appears to do better at classifying darts thrown near the center and the SVM model does better at classifying darts thrown away from the center. Let’s take a shot at stacking our models using Logistic Regression. We’ll use the base model predictions as meta features and DistFromCenter as an additional feature.

Sure enough the stacked model performs better than both of the base models – 75% CV accuracy and 86% test accuracy. Now let’s take a look at its classification regions overlaying the training data, just like we did with the base models.

The takeaway here is that the Logistic Regression Stacked Model captures the best aspects of each base model which is why it performs better than either base model in isolation.

### Stacking in Practice
To wrap this up, let’s talk about how, when, and why you might use stacking in the real world. Personally, I mostly use stacking in machine learning competitions on Kaggle. In general, stacking produces small gains with a lot of added complexity – not worth it for most businesses. But Stacking is almost always fruitful so it’s almost always used in top Kaggle solutions. In fact, stacking is really effective on Kaggle when you have a team of people trying to collaborate on a model. A single set of folds is agreed upon and then every team member builds their own model(s) using those folds. Then each model can be combined using a single stacking script. This is great because it prevents team members from stepping on each others toes, awkwardly trying to stitch their ideas into the same code base.

One last bit. Suppose we have dataset with (user, product) pairs and we want to predict the probability that a user will purchase a given product if he/she is presented an ad with that product. An effective feature might be something like, using the training data, what percent of the products advertised to a user did he actually purchase in the past? So, for the sample (user1, productA) in the training data, we want to tack on a feature like UserPurchasePercentage but we have to be careful not to introduce leakage into the data. We do this as follows:

Split the training data into folds
For each test fold
Identify the unique set of users in the test fold
Use the remaining folds to calculate UserPurchasePercentage (percent of advertised products each user purchased)
Map UserPurchasePercentage back to the training data via (fold id, user id)
Now we can use UserPurchasePercentage as a feature for our gradient boosting model (or whatever model we want). Effectively what we’ve just done is built a predictive model that predicts user_i will purchase product_x with probability based on the percent of advertised products he purchased in the past and used those predictions as a meta feature for our real model. This is a subtle but valid and effective form of stacking – one which I often do implement in practice and on Kaggle.
# Lesson 2

## Classification vs Regression

Classification and regression have very specific meanings.

Our `is_cat` classification function returns True or False. It's a categorization function used for a Classification 
model.

If you wanted to predict how old is a cat, or predict some kind of number, you call that Regression.

In other words: 
- Classification = Predicting from a discrete set of possibilities.
- Regression = Predicts one or more numeric quantities, such as temperature or location.

## Validation Data Set

When you train a model, you must always have a training set and a validation set, and must measure the accuracy of your
model on the validation set.

If you train for too long with not enough data, you will see accuracy getting worse due to over-fitting.

A learner is something that contains your data and your architecture to figure out which parameters best match the 
labels in your function.  
`resnet34` is an architecture that is very good at image recognition.

## Epoch, metrics, error rate and accuracy

`metrics=error_rate`, where you list the functions that you want to be called with your validation data and printed
out after each epoch.

epoch: Every time you look at every single image in the data set once.

Metric: A function that measures the quality of the model's predictions using the validation set, and will be printed
at the end of each epoch. `error_rate` is a function provided by fastai which tells you which images are being 
classified incorrectly. Another common metric is `accuracy`.

## Overfitting, training, validation and testing data set

Overfitting is the single most important and challenging issue when training models.
We don't want our model to get good results by cheating. It needs to be general at recognizing images.

Splitting off our validation data means our model never sees it in training, and so is completely untainted by it, 
right? Wrong. You can still cheat by fitting to the validation set. A test set, a third set of data, might be best.

## How to choose your training set

You don't want to choose from the middle. You want to choose from the end so it can predict the future.

* Understanding the test and validation sets is the most important thing to avoid failures.
* If considering an external vendor, hold out some test data to validate your performance. Don't tell the vendor.

## Transfer learning

Transfer Learning: Using a pretrained model for a task different to what it was originally trained for.

Ex: `resnet34` is good at image recognition trained on ImageNet. But if you then take that model and train it
on your model, you can end up with a far more accurate model. This is called transfer learning.

Transfer learning is a key technique to use less data and less compute and get better accuracy. A key focus for the 
fastAI library.

## Fine tuning

Fine Tuning: A transfer learning technique where the weights of a pretrained model are updated by training
for additional epochs using a different task to that used for pretraining.

## Why transfer learning works so well

Transfer learning works well because with each layer the model gets more sophisticated. It builds upon 
pre-learned features to find new features.

## Vision techniques used for sound

These techniques can also be used to recognize sounds by representing sounds as pictures.

## Using pictures to create fraud detection at Splunk

You can detect fraud by visualizing mouse movements as images and then creating an anti-fraud model.

## Detecting viruses using CNN

You can identify viruses by turning them into pictures.

## List of most important terms used in this course

See https://youtu.be/BvHmRx14HQ8?t=1892

## Arthur Samuel’s overall approach to neural networks

inputs + parameters => architecture -> predictions + labels -> loss -> update function -> parameters (loop).

## End of Chapter 1 of the Book

Chapter is over.

## Where to find pretrained models

You can find pretrained models at: 
* Google "model zoo" or "pretrained models".
* ImageNet is very general.
* Lots of opportunities for domain-specific models.

## The state of deep learning
## Recommendation vs Prediction
## Interpreting Models - P value
## Null Hypothesis Significance Testing
## Turn predictive model into something useful in production
## Practical exercise with Bing Image Search
## Bing Image Sign up
## Data Block API
## Summary

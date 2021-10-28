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
## How to choose your training set
## Transfer learning
## Fine tuning
## Why transfer learning works so well
## Vision techniques used for sound
## Using pictures to create fraud detection at Splunk
## Detecting viruses using CNN
## List of most important terms used in this course
## Arthur Samuel’s overall approach to neural networks
## End of Chapter 1 of the Book
## Where to find pretrained models
## The state of deep learning
## Recommendation vs Prediction
## Interpreting Models - P value
## Null Hypothesis Significance Testing
## Turn predictive model into something useful in production
## Practical exercise with Bing Image Search
## Bing Image Sign up
## Data Block API
## Summary

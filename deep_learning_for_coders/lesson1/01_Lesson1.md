# Lesson 1

Book: Deep Learning for Coders with fastai and PyTorch: AI Applications without a PhD, 1st Edition.

## What you don't need, to do deep learning

* Just high school math is sufficient.
* We've seen record-breaking results with <50 items of data.
* You can get what you need for state-of-the-art work for free.

## Where is deep learning the best-known approach?

* NLP
* Computer Vision
* Medicine
* Biology
* Image generation
* Recommendation systems
* Playing games
* Robotics
* Other applications

## Neural networks: a brief history

* Deep learning is based on neural network that started in 1943 with Warren McCulloch.

## The first AI Winter

People pointed out that AI models couldn't learn simple functions like XOR, even though the 
limitations could be addressed.

In 1986, Parallel Distributed Processing (PDP) was released that described an approach to 
deep learning.

## The age of deep learning

In the 80's, it was mathematically proven that the neural networks could work, but were often
too big and slow. We can now do this. **Deep** learning means **more layers**.

## Approach for learning

1. Play the whole game.
2. Make the game worth playing.
3. Work on the hard parts.

## The Software: PyTorch, fastai, and Jupyter (and why it doesn't matter)

* Python
* PyTorch (used to use Tensorflow but now it's too cumbersome)
* fastai

* PyTorch is designed for flexibility and developer-friendliness, but not beginner friendliness.
* fastai is the most popular higher-level API for PyTorch.

### Getting a GPU Deep learning server

You'll need an NVIDIA GPU (other GPUs are not supported by the main libraries).
Remember to shut down your instance (except for Colab).
Note that Colab doesn't automatically save your work.

### Jupyter Notebook

A thing where you can type Python code and shows you results.
Colab Notes: https://course.fast.ai/start_colab 

## Git Repositories

Fastbook repo: https://github.com/fastai/fastbook
* The full book with examples and images.

Course v4 repo: https://github.com/fastai/fastbook/tree/master/clean
* Only headings and code.

Colab: https://course.fast.ai/start_colab

## First practical example in Jupyter Notebook

Running on Colab since fastai is not supported on a Mac.

1. Grabbing a data set (Pets dataset)
2. Trying to figure out which ones are cats and dogs.
3. After a minute, it can do it with a minimal error rate.

## Interpretation and Explanation of the Exercise

Machine learning is a way to get computers to complete a specific task.

Normally, we write code like this:  
1. inputs -> program -> results.

Instead, machine learning is like this:
1. inputs + weights -> model -> results
2. Basic idea: Model is something that creates output not just on inputs, but also on some set of weights
3. Automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and
provide a mechanism for altering the weight assignment so as to maximize the performance. Then machine will "learn" 
from its experience.

So then: 
1. inputs + weights -> model -> results -> performance
2. Then update inputs and weights based on performance results.

We are building a computer program not by programming its individual steps, but through inference; by training it to 
learn to do the task.

Machine Learning:
```
The training of programs developed by allowing a computer to learn from its experience, rather than through 
manually coding the individual steps.
```

The model to be able to identify images is called a neural network. It's a very flexible function. A mathematical
proof called the `universal approximation theorem` shows that this function can solve any problem to any level of
accuracy, in theory.

## Stochastic Gradient Descent (SGD)

## Consider how a model interacts with its environment

## The "doc" function and fastai framework documentation

## Image Segmentation

## Classifying a review's sentiment based on IMDB text reviews

## Predicting a salary based on tabular data from CSV

## Summary

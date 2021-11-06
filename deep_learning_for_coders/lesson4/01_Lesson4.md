# Lesson 4

## Review of Lesson 3 + SGD

Arthur Samuel definition of Stochastic Gradient Descent.

## MNIST Loss Function

The view method lets you reshape things in PyTorch.

`train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)`
`train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)`

## What is a Dataset in PyTorch?

DataSet: A PyTorch concept that when indexed into it returns a tuple of independent and dependent variables.

```
dset = list(zip(train_x,train_y))
x,y = dset[0]
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

## Initializing our parameters
## Predicting images with matrix multiplication
## Why you shouldn't use accuracy loss function to update parameters
## Creating a good loss function
## Updating parameters with mini-batches and DataLoader
## Putting it all together
## Refactoring and Creating an optimizer
## The DataLoaders class
## The Learner class
## Adding a non-linearity to create a neural network
## Looking at what the NN is learning by looking at the parameters
## Comparing the results with the fastai toolkit
## Jargon review
## Is there a rule of thumb for which non-linearity to choose?
## Pet breeds image classification
## Presizing
## Checking and debugging a DataBlock
## Presizing (question)
## Training model to clean your data
## How fastai chooses a loss function
## Cross-Entropy Loss and Softmax
## Data Ethics and Efficacy of Masks for COVID-19

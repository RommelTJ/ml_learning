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

Random initialization.

`requires_grad_()` sets up the tensor to require a gradient.

`def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()`
`weights = init_params((28*28,1))`
`bias = init_params(1)`

Parameters: The weights and biases of a model.

## Predicting images with matrix multiplication

Prediction for one image:  
`(train_x[0]*weights.T).sum() + bias`

For-loop is slow because it wouldn't run on the GPU. Thus, we need to represent them in Higher-level functions.

We do this with matrix multiplication. In Python, matrix multiplication is represented with the `@` operator: 
```
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
```

## Why you shouldn't use accuracy loss function to update parameters

Because it gets  a 0 gradient all over the place.

## Creating a good loss function

What does a "slightly better prediction" look like?

```
trgts  = tensor([1,0,1])
prds   = tensor([0.9, 0.4, 0.2])

def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()
    
mnist_loss(prds,trgts) # tensor(0.4333)

mnist_loss(tensor([0.9, 0.4, 0.8]),trgts) # tensor(0.2333)
```

This is only going to look well as long as the predictions are always between 0 and 1.

We need a function to take numbers and turn them into numbers between 0 and 1.

Sigmoid function:  
`def sigmoid(x): return 1/(1+torch.exp(-x))`

```
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

## Updating parameters with mini-batches and DataLoader

Grab a few data items at a time to calculate loss and step. This is called a "mini-batch".

```
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
ds = L(enumerate(string.ascii_lowercase))
dl = DataLoader(ds, batch_size=6, shuffle=True)
```

## Putting it all together

Putting it all together... see `04_mnist_basics.ipynb`.

## Refactoring and Creating an optimizer

An optimizer is an object in PyTorch that will handle the SGD for us.

`linear_model = nn.Linear(28*28, 1)`

```
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```

fastai provides an SGD class which by default does the same thing as our BasicOptim:  
```
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
```

## The DataLoaders class

A class that accepts training and validation data.  
```
dls = DataLoaders(dl, valid_dl)
```

## The Learner class

A class that takes in a DataLoaders, model, optimization function, loss function, and optional metrics to print:  
```
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
```

Then you call `fit` and it trains our model:  
`learn.fit(10, lr=lr)`

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

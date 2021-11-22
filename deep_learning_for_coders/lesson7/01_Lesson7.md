# Lesson 7

## Weight decay (L2 Regularization)

Weight decay or L2 regularization is the process of changing the loss function to include the sum of all the weights
squared. The larger the coefficients, the sharper the canyons are.

```
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```

## Creating our own Embedding module

A computational shortcut for doing a matrix multiplication via one hot encoded matrix. The same as indexing into an 
array.

To treat tensors as parameters, you have to wrap it in the `nn.Parameter` class. `nn.Linear` is already doing it 
behind the scenes.
```
def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))
    
class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.movie_factors = create_params([n_movies, n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        movies = self.movie_factors[x[:,1]]
        res = (users*movies).sum(dim=1)
        res += self.user_bias[x[:,0]] + self.movie_bias[x[:,1]]
        return sigmoid_range(res, *self.y_range)

model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)
```

## Interpreting embeddings and bias
## Embedding distance
## Deep learning for collaborative filtering
## Notebook 9 - Tabular modelling
## entity embeddings for categorical variables
## Beyond deep learning for tabular data (ensembles of decision trees)
## Decision Trees
## Random Forests
## Out-of-bag error
## Model Interpretation
## extrapolation
## using a NN
## Ensembling
## Conclusion

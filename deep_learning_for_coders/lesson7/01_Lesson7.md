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

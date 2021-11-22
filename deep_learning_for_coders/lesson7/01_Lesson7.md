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

```
movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort()[:5]

idxs = movie_bias.argsort(descending=True)[:5]

g = ratings.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_idxs = tensor([learn.dls.classes['title'].o2i[m] for m in top_movies])
movie_w = learn.model.movie_factors[top_idxs].cpu().detach()
movie_pca = movie_w.pca(3)
fac0,fac1,fac2 = movie_pca.t()
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(12,12))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()
```

Using fastai.collab: 
```
learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
learn.fit_one_cycle(5, 5e-3, wd=0.1)

movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
```

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

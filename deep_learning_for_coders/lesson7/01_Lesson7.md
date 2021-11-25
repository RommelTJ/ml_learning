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

We can find the embedding distances between movies. This helps us find the similarity between movies.

```
movie_factors = learn.model.i_weight.weight
idx = dls.classes['title'].o2i['Silence of the Lambs, The (1991)']
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distances.argsort(descending=True)[1]
dls.classes['title'][idx]
```

## Deep learning for collaborative filtering

```
class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)
        
model = CollabNN(*embs)

learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.01)

learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)

@delegates(TabularModel)
class EmbeddingNN(TabularModel):
    def __init__(self, emb_szs, layers, **kwargs):
        super().__init__(emb_szs, layers=layers, n_cont=0, out_sz=1, **kwargs)
```

## Notebook 9 - Tabular modelling

Embeddings for any kind of categorical variable. It can handle any kind of discrete categorical data. 
Something like "sex", "post code", etc. Something with cardinality.

## Entity embeddings for categorical variables

Kaggle competitions to determine embedding distance. Recommendation system in Google Play.

## Beyond deep learning for tabular data (ensembles of decision trees)

The idea of deep learning as a best practice for tabular data is new and controversial. Without it, we would be using 
an ensemble of decision trees (Random Forests or Gradient Boosting Machines).

Ensemble of Decision Trees are easier to interpret, do not require GPU hardware, and require less hyperparameter tuning.
They should be our first approach for analysing a new tabular dataset, except when there is too high cardinality 
variables or when there are columns which contain data which would be best understood with a neural network, such
as plaintext data.

PyTorch is bad for this. Use SciKit-Learn (sklearn).

We'll use the "Blue Book for Bulldozers" dataset from a Kaggle competition.

```
creds = 'YOUR_KAGGLE_CREDS'

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)
    
path = URLs.path('bluebook')
Path.BASE_PATH = path

if not path.exists():
    path.mkdir(parents=true)
    api.competition_download_cli('bluebook-for-bulldozers', path=path)
    file_extract(path/'bluebook-for-bulldozers.zip')

path.ls(file_type='text')
```

```
df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
df.columns
df['ProductSize'].unique()
sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'
df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)
dep_var = 'SalePrice'
df[dep_var] = np.log(df[dep_var])
```

## Decision Trees

A decision tree asks a series of binary questions about data.

Handling Dates
```
df = add_datepart(df, 'saledate')
df_test = pd.read_csv(path/'Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')
' '.join(o for o in df.columns if o.startswith('sale'))
```

Using TabularPandas and TabularProc
```
procs = [Categorify, FillMissing]

cond = (df.saleYear<2011) | (df.saleMonth<10)
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))

cont,cat = cont_cat_split(df, 1, dep_var=dep_var)

to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
len(to.train),len(to.valid)

to.show(3)

to1 = TabularPandas(df, procs, ['state', 'ProductGroup', 'Drive_System', 'Enclosure'], [], y_names=dep_var, splits=splits)
to1.show(3)

to.items.head(3)

to1.items[['state', 'ProductGroup', 'Drive_System', 'Enclosure']].head(3)

to.classes['ProductSize']

save_pickle(path/'to.pkl',to)
```

Creating the Decision Tree
```
to = load_pickle(path/'to.pkl')

xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y

m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);

draw_tree(m, xs, size=10, leaves_parallel=True, precision=2)

samp_idx = np.random.permutation(len(y))[:500]
dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')

xs.loc[xs['YearMade']<1900, 'YearMade'] = 1950
valid_xs.loc[valid_xs['YearMade']<1900, 'YearMade'] = 1950

m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)

dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')

m = DecisionTreeRegressor()
m.fit(xs, y);

def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)

m_rmse(m, xs, y)
m_rmse(m, valid_xs, valid_y)
m.get_n_leaves(), len(xs)

m = DecisionTreeRegressor(min_samples_leaf=25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)

m.get_n_leaves()
```

## Random Forests

Bagging predictors is a method for generating multiple versions of a predictor and using these to get an aggregated 
predictor. 

Creating a Random Forest:
```
def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
        
m = rf(xs, y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
r_mse(preds.mean(0), valid_y)
plt.plot([r_mse(preds[:i+1].mean(0), valid_y) for i in range(40)]);
```

## Out-of-bag error
## Model Interpretation
## Extrapolation
## Using a NN
## Ensembling
## Conclusion

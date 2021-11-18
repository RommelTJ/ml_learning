# Lesson 6

## Pet Breeds

Continuing on training CNN for computer vision. Training image classifier for breeds of pets.

Please review Cross entropy loss. If confused, review last lesson or `mnist loss` as its similar.

## Model Interpretation

We can use a confusion matrix. 
* Diagonal shows classified correctly.

When you have many classes, a better thing might be to use: 
* `interp.most_confused(min_val=5)`
* Gives you the most incorrect predictions.

## Improving our model

One way is to improve the learning rate. You can do this by calling the `fine_tune` method.
```
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1, base_lr=0.1)
```

You can use a learning rate finder to find a good learning rate.
```
learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min,lr_steep = learn.lr_find()
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2, base_lr=3e-3)
```
Now we have 8.3% error rate after 3 epochs. Learning Rate Finder was invented in 2015. 

Transfer learning is to take a pretrained model and fine tune it for some other task.
We take a model, set of parameters, throw away the last layer, and replace it with random weights, and train that.

`freeze` is the method so that only the last layer's steps get optimized.
`fit` is fitting those randomly added weights.

We can optimize further by having a small learning rate in early layers, and a large learning rate for later layers.
This is something called "Discriminative learning rates".
```
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
learn.recorder.plot_loss()
```
Now we have less than 5.5% error rate.

Deeper architectures adds more pairs of activation functions. 34 means 34 pairs.
Using a deeper model requires more GPU RAM. They also take longer to train.

Mixed precision training uses less precise numbers to speed up when possible during training.
NVIDIA GPUs have "tensor cores" which can dramatically speed up things.
But it doesn't always get better! Using resnet50 resulted in the same 5.5% error rate.

## Multi-label classification

Example: Images with multiple bicycles, car, persons.
```
from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(path/'train.csv')
df.head()
```

Pandas is a library used to create data for DataFrames (in our case). 
* You can assess rows and columns of a DataFrame with the `iloc` property
  * `df.iloc[:,0]` means every row, 0th column.
  * `df.iloc[0,:]` means 0th row, every column.
* You can grab a column by name by indexing into a DataFrame
  * `df['fname']`
* You can create new columns and do calculations using columns
  * `tmp_df = pd.DataFrame({'a':[1,2], 'b':[3,4]})` creates a new DataFrame
  * `tmp_df['c'] = tmp_df['a']+tmp_df['b']` is adding two columns.
* It has a confusing API. Python for Data Analysis by Wes McKinney.


Constructing a DataBlock
```
dblock = DataBlock()
dsets = dblock.datasets(df)
len(dsets.train),len(dsets.valid)
dblock = DataBlock(get_x = lambda r: r['fname'], get_y = lambda r: r['labels'])
dsets = dblock.datasets(df)
dsets.train[0]

def get_x(r): return r['fname']
def get_y(r): return r['labels']
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]

def get_x(r): return path/'train'/r['fname']
def get_y(r): return r['labels'].split(' ')
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]

idxs = torch.where(dsets.train[0][1]==1.)[0]
dsets.train.vocab[idxs]

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y)

dsets = dblock.datasets(df)
dsets.train[0]

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms = RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(df)
dls.show_batch(nrows=1, ncols=3)
```

* Dataset
  * A collection which returns a tuple of your independent and dependent variable for a single item
* Datasets
  * An object which contains a training Dataset and a validation Dataset
* DataLoader
  * An iterator which provides a stream of mini batches, where each mini batch is a couple of a batch of independent 
    variables and a batch of dependent variables.

Python Tip:
* Shortcut in zip: `zip(*b)`, means insert into zip every element of variable b.
* In practice, it's used to transpose something from one orientation to another.

## One hot encoding

Since we have multiple categories, there will be zeros in most categories, and 1s for the present ones. This is called
"one hot encoding" and it allows us to use tensors which require items to be of the same length.

* One hot encoding
  * Using a vector of zeros, with a one in each location that is represented in the data, to encode a list of integers.

```
idxs = torch.where(dsets.train[0][1]==1.)[0]
dsets.train.vocab[idxs]
```

```
def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y)

dsets = dblock.datasets(df)
dsets.train[0]
```

```
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms = RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(df)
dls.show_batch(nrows=1, ncols=3)
```

Binary cross entropy
* Create a learner
  * `learn = cnn_learner(dls, resnet18)`
* Train batch of data
  * `x,y = to_cpu(dls.train.one_batch())`
  * `activs = learn.model(x)`
  * `activs.shape`
* Define binary cross entropy
```
def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets==1, inputs, 1-inputs).log().mean()
```
  * Identical to mnist loss but takes log and mean.
  * Or using fastAI
```
loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
```

```
learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)

learn.metrics = partial(accuracy_multi, thresh=0.1)
learn.validate()

learn.metrics = partial(accuracy_multi, thresh=0.99)
learn.validate()

preds,targs = learn.get_preds()
accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)

xs = torch.linspace(0.05,0.95,29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
plt.plot(xs,accs);
```

## Regression

Image regression is when our dependent variable is a continuous number.
Ex: Find the mid-point of a person's head.
Ex: Biwi Head Pose dataset.

```
path = untar_data(URLs.BIWI_HEAD_POSE)
Path.BASE_PATH = path
img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
im = PILImage.create(img_files[0])

cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])
get_ctr(img_files[0])

biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms=[*aug_transforms(size=(240,320)), 
                Normalize.from_stats(*imagenet_stats)]
)
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
xb,yb = dls.one_batch()
```

Training a model
```
learn = cnn_learner(dls, resnet18, y_range=(-1,1))

def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo
plot_function(partial(sigmoid_range,lo=-1,hi=1), min=-4, max=4)

dls.loss_func
learn.lr_find()
lr = 1e-2
learn.fine_tune(3, lr)
learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
```

dls learns using a sigmoid range function.

We fine-tuned an image classification model to an image regression model.

## Collaborative Filtering

We can recommend movies if we know what you liked in the past. These can be figured out via concepts like what 
kind of movie it is and so on. These are called latent factors. Nobody necessarily wrote them down or communicated them.

Using MovieLens data set.

Outside of recommender systems, collaborative filtering might be used whenever you want to learn from past behavior.

```
from fastai.collab import *
from fastai.tabular.all import *
path = untar_data(URLs.ML_100k)

ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names=['user','movie','rating','timestamp'])
last_skywalker = np.array([0.98,0.9,-0.9])
user1 = np.array([0.9,0.8,-0.6])
(user1*last_skywalker).sum()
casablanca = np.array([-0.99,-0.3,0.8])
(user1*casablanca).sum()

movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1', 
                     usecols=(0,1), names=('movie','title'), header=None)
ratings = ratings.merge(movies)
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.classes

n_users  = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)

one_hot_3 = one_hot(3, n_users).float()
user_factors.t() @ one_hot_3
user_factors[3]
```

## Embedding

Embedding layer
* Multiplying by a one hot encoded matrix, using the computational shortcut that it can be implemented by simply 
  indexing directly.
* The thing that you multiply the one hot encoded matrix by (or, using the computation shortcut, index into directly)
  is called the embedding matrix.

Most deep learning libraries, including PyTorch, support embedding.

## Collaborative filtering from scratch

```
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return (users * movies).sum(dim=1)
        
x,y = dls.one_batch()
model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```

```
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return sigmoid_range((users * movies).sum(dim=1), *self.y_range)

model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```

```
class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)

model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```

## Regularisation (Data augmentation for regression)

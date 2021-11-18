# Questionnaire Answers

1. Why can't we use torch.where to create a loss function for datasets where our label can have more than two categories?

Because `torch.where` can only select between two possibilities while for multi-class classification, we have 
multiple possibilities.

2. What is the value of log(-2)? Why? 

This value is not defined. The logarithm is the inverse of the exponential function, and the exponential function is 
always positive no matter what value is passed. So the logarithm is not defined for negative values.

3. What are two good rules of thumb for picking a learning rate from the learning rate finder?

Either one of these two points should be selected for the learning rate:
* One order of magnitude less than where the minimum loss was achieved (i.e. the minimum divided by 10)
* The last point where the loss was clearly decreasing.

4. What two steps does the fine_tune method do?

* Train the new head (with random weights) for one epoch
* Unfreeze all the layers and train them all for the requested number of epochs

5. In Jupyter Notebook, how do you get the source code for a method or function?

Use ?? after the function ex: DataBlock.summary??

6. What are discriminative learning rates?

Discriminative learning rates refers to the training trick of using different learning rates for different layers of 
the model. This is commonly used in transfer learning. The idea is that when you train a pretrained model, you don't 
want to drastically change the earlier layers as it contains information regarding simple features like edges and 
shapes. But later layers may be changed a little more as it may contain information regarding facial feature or 
other object features that may not be relevant to your task. Therefore, the earlier layers have a lower learning rate 
and the later layers have higher learning rates.

7. How is a Python slice object interpreted when passed as a learning rate to fastai?

The first value of the slice object is the learning rate for the earliest layer, while the second value is the 
learning rate for the last layer. The layers in between will have learning rates that are multiplicatively 
equidistant throughout that range.

8. Why is early stopping a poor choice when using 1cycle training?

If early stopping is used, the training may not have time to reach lower learning rate values in the learning 
rate schedule, which could easily continue to improve the model. Therefore, it is recommended to retrain the model 
from scratch and select the number of epochs based on where the previous best results were found.

9. What is the difference between resnet50 and resnet101?

The number 50 and 101 refer to the number of layers in the models. Therefore, ResNet101 is a larger model with more 
layers versus ResNet50. These model variants are commonly as there are ImageNet-pretrained weights available.

10. What does to_fp16 do?

This enables mixed-precision training, in which less precise numbers are used in order to speed up training.

11. How could multi-label classification improve the usability of the bear classifier?

This would allow for the classification of no bears present. Otherwise, a mutli-class classification model will 
predict the presence of a bear even if it’s not there (unless a separate class is explicitly added).

12. How do we encode the dependent variable in a multi-label classification problem?

This is encoded as a one-hot encoded vector. Essentially, this means we have a zero vector of the same length of the 
number of classes, and ones are present at the indices for the classes that are present in the data.

13. How do you access the rows and columns of a DataFrame as if it was a matrix?

You can use .iloc. For example, df.iloc[10,10] will select the element in the 10th row and 10th column as if the 
DataFrame is a matrix.

14. How do you get a column by name from a DataFrame?

This is very simple. You can just index it! Ex: df['column_name']

15. What is the difference between a Dataset and DataLoader?

A Dataset is a collection which returns a tuple of your independent and dependent variable for a single item. 

A DataLoader is an extension of the Dataset functionality. It is an iterator which provides a stream of mini batches, 
where each mini batch is a couple of a batch of independent variables and a batch of dependent variables.

16. What does a Datasets object normally contain?

A training set and validation set.

17. What does a DataLoaders object normally contain?

A trainin dataloader and validation dataloader.

18. What does lambda do in Python?

Lambdas are shortcuts for writing functions (writing one-liner functions). It is great for quick prototyping and 
iterating, but since it is not serializable, it cannot be used in deployment and production.

19. What are the methods to customize how the independent and dependent variables are created with the data block API?

get_x and get_y
* get_x is used to specify how the independent variables are created.
* get_y is used to specify how the datapoints are labelled

20. Why is softmax not an appropriate output activation function when using a one hot encoded target?

Softmax wants to make the model predict only a single class, which may not be true in a multi-label classification 
problem. In multi-label classification problems, the input data could have multiple labels or even no labels.

21. Why is nll_loss not an appropriate loss function when using a one-hot-encoded target?

Again, nll_loss only works for when the model only needs to predict one class, which is not the case here.

22. What is the difference between nn.BCELoss and nn.BCEWithLogitsLoss?

nn.BCELoss does not include the initial sigmoid. It assumes that the appropriate activation function (i.e. the sigmoid) 
has already been applied to the predictions. 

nn.BCEWithLogitsLoss, on the other hand, does both the sigmoid and cross entropy in a single function.

23. Why can't we use regular accuracy in a multi-label problem?

The regular accuracy function assumes that the final model-predicted class is the one with the highest activation. 
However, in multi-label problems, there can be multiple labels. Therefore, a threshold for the activations needs to 
be set for choosing the final predicted classes based on the activations, for comparing to the target classes.

24. When is it okay to tune a hyperparameter on the validation set?

It is okay to do so when the relationship between the hyper-parameter and the metric being observed is smooth. 
With such a smooth relationship, we would not be picking an inappropriate outlier.

25. How is y_range implemented in fastai? (See if you can implement it yourself and test it without peeking!)

y_range is implemented using sigmoid_range in fastai.
 
def sigmoid_range(x, lo, hi): return x.sigmoid() * (hi-lo) + lo

26. What is a regression problem? What loss function should you use for such a problem?

In a regression problem, the dependent variable or labels we are trying to predict are continuous values. For such 
problems, the mean squared error loss function is used.

27. What do you need to do to make sure the fastai library applies the same data augmentation to your inputs images 
and your target point coordinates?

You need to use the correct DataBlock. In this case, it is the PointBlock. 

This DataBlock automatically handles the application data augmentation to the input images and the target point 
coordinates.

28. What problem does collaborative filtering solve?

It solves the problem of predicting the interests of users based on the interests of other users and recommending 
items based on these interests.

29. How does it solve it?

The key idea of collaborative filtering is latent factors. The idea is that the model can tell what kind of items 
you may like (ex: you like sci-fi movies/books) and these kinds of factors are learned (via basic gradient descent) 
based on what items other users like.

30. Why might a collaborative filtering predictive model fail to be a very useful recommendation system?

If there are not many recommendations to learn from, or enough data about the user to provide useful recommendations, 
then such collaborative filtering systems may not be useful.

31. What does a crosstab representation of collaborative filtering data look like?

In the crosstab representation, the users and items are the rows and columns (or vice versa) of a large matrix with 
the values filled out based on the user’s rating of the item.

32. Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching!).

Ok.

33. What is a latent factor? Why is it "latent"?

As described above, a latent factor are factors that are important for the prediction of the recommendations, but are 
not explicitly given to the model and instead learned (hence "latent").

34. What is a dot product? Calculate a dot product manually using pure Python with lists.

A dot product is when you multiply the corresponding elements of two vectors and add them up. If we represent the 
vectors as lists of the same size, here is how we can perform a dot product:


a = [1, 2, 3, 4] 
b = [5, 6, 7, 8] 
dot_product = sum(i[0]*i[1] for i in zip(a,b))

35. What does pandas.DataFrame.merge do?

It allows you to merge DataFrames into one DataFrame.

36. What is an embedding matrix?

It is what you multiply an embedding with, and in the case of this collaborative filtering problem, is 
learned through training.

37. What is the relationship between an embedding and a matrix of one-hot-encoded vectors?

An embedding is a matrix of one-hot encoded vectors that is computationally more efficient.

38. Why do we need Embedding if we could use one-hot-encoded vectors for the same thing?

Embedding is computationally more efficient. The multiplication with one-hot encoded vectors is equivalent to 
indexing into the embedding matrix, and the Embedding layer does this. However, the gradient is calculated such that 
it is equivalent to the multiplication with the one-hot encoded vectors.

39. What does an embedding contain before we start training (assuming we're not using a pretained model)?

The embedding is randomly initialized.

40. Create a class (without peeking, if possible!) and use it.

```
class Example:
    def __init__(self, a): self.a = a
    def say(self,x): return f'Hello {self.a}, {x}.'
```

41. What does x[:,0] return?

The user ids.

42. Rewrite the DotProduct class (without peeking, if possible!) and train a model with it.

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
```

43. What is a good loss function to use for MovieLens? Why?

We can use Mean Squared Error (MSE), which is a perfectly reasonable loss as we have numerical targets for the 
ratings and it is one possible way of representing the accuracy of the model.

44. What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?

We would need to ensure the model outputs 5 predictions. For example, with a neural network model, we need to change 
the last linear layer to output 5, not 1, predictions. Then this is passed into the Cross Entropy loss.

45. What is the use of bias in a dot product model?

A bias will compensate for the fact that some movies are just amazing or pretty bad. It will also compensate for 
users who often have more positive or negative recommendations in general.

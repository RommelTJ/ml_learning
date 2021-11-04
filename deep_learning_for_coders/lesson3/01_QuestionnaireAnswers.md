# Questionnaire Answers

1. What letters are often used to signify the independent and dependent variables?

x is independent and y is dependent.

2. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?

Crop is the default. It crops to the desired shape and might lose important details.  
Pad padds the image with bars. It preserves the shape, but it may lead to wasted computation and poorer resolution.  
Squish streches the image to fit. It may lead the model to learn that things look differently than they are.

3. What is data augmentation? Why is it needed?

Data augmentation refers to creating random variations of our input data, such that they appear different, but not 
so different that it changes the meaning of the data. Examples include flipping, rotation, perspective warping, 
brightness changes, etc. Data augmentation is useful for the model to better understand the basic concept of what 
an object is and how the objects of interest are represented in images. Therefore, data augmentation allows machine 
learning models to generalize . This is especially important when it can be slow and expensive to label data.

4. What is the difference between item_tfms and batch_tfms?

`item_tfms` are transformations applied to a single data sample x on the CPU. Resize() is a common transform because 
the mini-batch of input images to a cnn must have the same dimensions. Assuming the images are RGB with 3 channels, 
then Resize() as `item_tfms` will make sure the images have the same width and height.

`batch_tfms` are applied to batched data samples (aka individual samples that have been collated into a mini-batch) 
on the GPU. They are faster and more efficient than item_tfms. A good example of these are the ones provided by 
aug_transforms(). Inside are several batch-level augmentations that help many models.

5. What is a confusion matrix?

A class confusion matrix is a representation of the predictions made vs the correct labels. The rows of the matrix 
represent the actual labels while the columns represent the predictions. Therefore, the number of images in the 
diagonal elements represent the number of correctly classified images, while the off-diagonal elements are 
incorrectly classified images. Class confusion matrices provide useful information about how well the model is 
doing and which classes the model might be confusing.

6. What does export save?

export saves both the architecture, as well as the trained parameters of the neural network architecture. It also 
saves how the DataLoaders are defined.

7. What is it called when we use a model for getting predictions, instead of training?

Inference

8. What are IPython widgets?

IPython widgets are JavaScript and Python combined functionalities that let us build and interact with GUI 
components directly in a Jupyter notebook. An example of this would be an upload button, which can be created with 
the Python function widgets.FileUpload().

9. When might you want to use CPU for deployment? When might GPU be better?

GPUs are best for doing identical work in parallel. If you will be analyzing single pieces of data at a time (like a 
single image or single sentence), then CPUs may be more cost effective instead, especially with more market 
competition for CPU servers versus GPU servers. GPUs could be used if you collect user responses into a batch at a 
time, and perform inference on the batch. This may require the user to wait for model predictions. Additionally, 
there are many other complexities when it comes to GPU inference, like memory management and queuing of the batches.

10. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?

The application will require network connection, and there will be extra network latency time when submitting input 
and returning results. Additionally, sending private data to a network server can lead to security concerns.

On the flip side deploying a model to a server makes it easier to iterate and roll out new versions of a model. This 
is because you as a developer have full control over the server environment and only need to do it once rather 
than having to make sure that all the endpoints (phones, PCs) upgrade their version individually.

11. What are three examples of problems that could occur when rolling out a bear warning system in practice?

The model we trained will likely perform poorly when:
* Handling night-time images
* Dealing with low-resolution images (ex: some smartphone images)
* The model returns prediction too slowly to be useful

12. What is "out-of-domain data"?

Data that is fundamentally different in some aspect compared to the model's training data. For example, an object 
detector that was trained exclusively with outside daytime photos is given a photo taken at night.

13. What is "domain shift"?

This is when the type of data changes gradually over time. For example, an insurance company is using a deep 
learning model as part of their pricing algorithm, but over time their customers will be different, with the 
original training data not being representative of current data, and the deep learning model being applied on 
effectively out-of-domain data.

14. What are the three steps in the deployment process?

* Manual process
* Limited scope deployment
* Gradual expansion

15. How is a grayscale image represented on a computer? How about a color image?

Images are represented by arrays with pixel values representing the content of the image. 

For greyscale images, a 2-dimensional array is used with the pixels representing the greyscale values, with a 
range of 256 integers. A value of 0 would represent white, and a value of 255 represents black, and different 
shades of greyscale in between. 

For color images, three color channels (red, green, blue) are typically used, with a separate 256-range 2D array 
used for each channel. A pixel value of 0 again represents white, with 255 representing solid red, green, or blue. 
The three 2-D arrays form a final 3-D array (rank 3 tensor) representing the color image.

16. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?

There are two subfolders, train and valid, the former contains the data for model training, the latter contains 
the data for validating model performance after each training step. Evaluating the model on the validation set 
serves two purposes: a) to report a human-interpretable metric such as accuracy (in contrast to the often abstract 
loss functions used for training), b) to facilitate the detection of overfitting by evaluating the model on a 
dataset it hasn't been trained on (in short, an overfitting model performs increasingly well on the training set 
but decreasingly so on the validation set). Of course, every practitioner could generate their own 
train/validation-split of the data. Public datasets are usually pre-split to simplify comparing results 
between implementations/publications.

Each subfolder has two subfolders 3 and 7 which contain the .jpg files for the respective class of images. 
This is a common way of organizing datasets comprised of pictures. For the full MNIST dataset there are 10 subfolders, 
one for the images for each digit.

17. Explain how the "pixel similarity" approach to classifying digits works.

We generate an archetype for each class we want to identify. In our case, we want to distinguish images of 3's from 
images of 7's. 

We define the archetypal 3 as the pixel-wise mean value of all 3's in the training set. You can visualize the two 
archetypes and see that they are in fact blurred versions of the numbers they represent.

In order to tell if a previously unseen image is a 3 or a 7, we calculate its distance to the two archetypes 
(here: mean pixel-wise absolute difference). We say the new image is a 3 if its distance to the archetypal 3 is lower 
than two the archetypal 7.

18. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.

A list comprehension is a Pythonic way of condensing the creation of a list using a for-loop into a single expression. 
List comprehensions will also often include if clauses for filtering.

19. What is a "rank-3 tensor"?

A tensor with 3 dimensions.

20. What is the difference between tensor rank and shape? How do you get the rank from the shape?

Rank is the number of dimensions in a tensor; shape is the size of each dimension of a tensor.

21. What are RMSE and L1 norm?

Root mean square error (RMSE), also called the L2 norm, and mean absolute difference (MAE), also called the L1 norm, 
are two commonly used methods of measuring "distance". Simple differences do not work because some differences are 
positive and others are negative, canceling each other out. Therefore, a function that focuses on the magnitudes of 
the differences is needed to properly measure distances. The simplest would be to add the absolute values of 
the differences, which is what MAE is. RMSE takes the mean of the square (makes everything positive) and then 
takes the square root (undoes squaring).

22. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

As loops are very slow in Python, it is best to represent the operations as array operations rather than looping 
through individual elements. If this can be done, then using NumPy or PyTorch will be thousands of times faster, as 
they use underlying C code which is much faster than pure Python. Even better, PyTorch allows you to run operations 
on GPU, which will have significant speedup if there are parallel operations that can be done.

23. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.

```
torch.Tensor(list(range(1,10))).view(3,3)
tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

b = 2*a
tensor([[ 2.,  4.,  6.], [ 8., 10., 12.], [14., 16., 18.]])

b[1:,1:]
tensor([[10., 12.], [16., 18.]])
```

24. What is broadcasting?

Scientific/numerical Python packages like NumPy and PyTorch will often implement broadcasting that often makes code 
easier to write. In the case of PyTorch, tensors with smaller rank are expanded to have the same size as the larger 
rank tensor. In this way, operations can be performed between tensors with different rank.

25. Are metrics generally calculated using the training set, or the validation set? Why?

Metrics are generally calculated on a validation set. As the validation set is unseen data for the model, evaluating 
the metrics on the validation set is better in order to determine if there is any overfitting and how well the model 
might generalize if given similar data.

26. What is SGD?

SGD, or stochastic gradient descent, is an optimization algorithm. Specifically, SGD is an algorithm that will update 
the parameters of a model in order to minimize a given loss function that was evaluated on the predictions and target. 

The key idea behind SGD (and many optimization algorithms, for that matter) is that the gradient of the loss function 
provides an indication of how that loss function changes in the parameter space, which we can use to determine how 
best to update the parameters in order to minimize the loss function. This is what SGD does.

27. Why does SGD use mini-batches?

We need to calculate our loss function (and our gradient) on one or more data points. We cannot calculate on the whole 
datasets due to compute limitations and time constraints. If we iterated through each data point, however, the 
gradient will be unstable and imprecise, and is not suitable for training. As a compromise, we calculate the 
average loss for a small subset of the dataset at a time. This subset is called a mini-batch. Using mini-batches are 
also more computationally efficient than single items on a GPU.

28. What are the seven steps in SGD for machine learning?

* Initialize the parameters – Random values often work best.
* Calculate the predictions – This is done on the training set, one mini-batch at a time.
* Calculate the loss – The average loss over the minibatch is calculated
* Calculate the gradients – this is an approximation of how the parameters need to change in order to minimize the loss function
* Step the weights – update the parameters based on the calculated weights
* Repeat the process
* Stop – In practice, this is either based on time constraints or usually based on when the training/validation losses and metrics stop improving.

29. How do we initialize the weights in a model?

Random weights work pretty well.

30. What is "loss"?

The loss function will return a value based on the given predictions and targets, where lower values correspond to 
better model predictions.

31. Why can't we always use a high learning rate?

The loss may "bounce" around (oscillate) or even diverge, as the optimizer is taking steps that are too large, and 
updating the parameters faster than it should be.

32. What is a "gradient"?

The gradients tell us how much we have to change each weight to make our model better. It is essentially a measure of 
how the loss function changes with changes of the weights of the model (the derivative).

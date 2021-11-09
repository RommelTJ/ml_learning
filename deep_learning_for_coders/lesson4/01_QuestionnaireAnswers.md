# Questionnaire Answers

1. Why can't we use accuracy as a loss function?

Accuracy isn't good to use as a loss function because it only changes when the predictions of the model change. It 
can improve the confidence of its predictions, but unless the predictions actually change, the accuracy will 
remain the same. It also produces gradients that are mostly equal to zero which prevents the parameters from 
updating during the training process.

2. Draw the sigmoid function. What is special about its shape?

The sigmoid function is an activation function that's named after its shape which resembles the letter “S” when 
plotted. It has a smooth curve that gradually transitions from values above 0.0 to values just below 1.0. It also 
only goes up which makes it easier for SGD to find meaningful gradients.

3. What is the difference between a loss function and a metric?

The loss function is used to evaluate and diagnose how well the model is learning during the optimization step of 
the training process. It responds to small changes in confidence levels which helps to minimize the loss and monitor 
for things like overfitting, underfitting, and convergence. It also gets calculated for each item in the dataset, 
and at the end of each epoch where the loss values are all averaged and the overall mean is reported.

The metric is used to evaluate the model and perform model selection during the evaluation process after the 
training process. It provides an interpretation of the performance of the model that’s easier for humans to 
understand which helps give meaning to the performance in the context of the goals of the overall project and 
project stakeholders. It also gets printed at the end of each epoch which reports the performance of the model.

4. What is the function to calculate new weights using a learning rate?

The Optimizer is an optimization algorithm that's used in machine learning to update the weights based on the 
gradients during the optimization step of the training process. It starts by defining some kind of loss function 
and ends by minimizing the loss using one of the optimization routines. It can also make the difference between 
getting a good accuracy in hours or days.

5. What does the DataLoader class do?

The DataLoader is a class that's used in PyTorch to preprocess the dataset into the format that's expected by the 
model. It specifies the dataset to load, randomly shuffles the dataset, creates the mini-batches, and loads the 
mini-batches in parallel. It also returns a dataloader object that contains tuples of tensors that represent the 
batches of independent and dependent variables.

6. Write pseudocode showing the basic steps taken in each epoch for SGD.

See `04_mnist_basics.ipynb`.

7. Create a function that, if passed two arguments [1,2,3,4] and 'abcd', 
returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?

The output is special because it has the same data structure as the Dataset object that's used in PyTorch. It 
contains a list of tuples where each tuple stores an item with the associated label. It also contains all the 
items and labels from the first and second parameters which are paired at each index.

8. What does view do in PyTorch?

The View is a method that's used in PyTorch to reshape the tensor without changing its contents. It doesn't create 
a copy of the data which allows for efficient memory-efficient reshaping, slicing, and element-wise operations. 
It also shares the underlying data with the original tensor which means any changes made to the data in the view 
will be reflected in the original tensor.

9. What are the "bias" parameters in a neural network? Why do we need them?

The Bias is a parameter that's used in machine learning to offset the output inside the model to better fit the 
data during the training process. It shifts the activation function to the left or right which moves the entire 
curve to delay or accelerate the activation. It also gets added to the product of the inputs and weights before 
being passed through the activation function.

`parameters = sum(inputs * weights) + bias`

10. What does the @ operator do in Python?

The @ is an operator that's used in Python to perform matrix multiplication between two arrays. It performs the 
same operation as the matmul function from the NumPy library. It also makes matrix formulas much easier to read 
which makes it much easier to work with for both experts and non-experts.

```
np.matmul(np.matmul(np.matmul(A, B), C), D)
A @ B @ C @ D
```

11. What does the backward method do?

Backward is a method that's used in PyTorch to calculate the gradient of the loss. It performs the 
backpropagation using the backward method in the Tensor class from the PyTorch library. It also adds the 
gradients to any other gradients that are currently stored in the grad attribute in the tensor object.

12. Why do we have to zero the gradients?

In PyTorch, the gradients accumulate on subsequent backward passes by default. It helps train recurrent neural 
networks that work with time-series data where the backpropagation is repeated to perform backpropagation 
through time. It also must be manually set to zero for most neural networks before the backward pass is performed 
to update the parameters correctly.

```
learning_rate = 1e-5
parameters.data -= learning_rate * parameters.grad.data
parameters.grad = None
```

13. What information do we have to pass to Learner?

The Learner is a class that's used in Fastai to train the model. It specifies the data loaders and model objects 
that are required to train the model and perform transfer learning. It can also specify the optimizer function, 
loss function, and other optional parameters that already have default values.

```
learner = Learner(dataloaders, model, loss_function, optimizer_function, metrics)
```

14. Show Python or pseudocode for the basic steps of a training loop.

```
for _ in range(epochs):
    prediction = model(x_batch, parameters)
    loss = loss(prediction, label)
    loss.backward()
    for parameterin parameters:
        parameter.grad.data += learning_rate * parameter.grad.data
        parameter.grad.data = None
```

15. What is "ReLU"? Draw a plot of it for values from -2 to +2.

Rectified Linear Unit (ReLU) is an activation function that's used in machine learning to address the vanishing 
gradient problem. It activates the input value for all the positive values and replaces all the negative values 
with zero. It also decreases the ability of the model to train properly when there are too many activations as 
zero because the gradient of zero is zero which prevents those parameters from being updated during the backward pass.

16. What is an "activation function"?

The Activation Function is a function that's used in machine learning to decide whether the input is relevant or 
irrelevant. It gets attached to each neuron in the artificial network and determines whether to activate based 
on whether the input is relevant for the prediction of the model. It also helps normalize the output of each 
neuron to a range between -1 and 1.

`output = activation_function(parameters)`

17. What's the difference between F.relu and nn.ReLU?

F.relu is a function that's used in PyTorch to apply the rectified linear unit function to the layers in the 
model that's manually defined in the class. It must be manually defined in the class of the artificial neural 
network where the layers and functions are defined as class attributes. It also does the same thing as the 
nn.ReLU class which builds the model with sequential modules.

nn.ReLU is a class that's used in PyTorch to apply the rectified linear unit function to the layers in the model 
that's defined using sequential modules. It must be used with other sequential modules which represent the layers 
and functions that build the artificial neural network. It also does the same thing as the F.relu function which 
builds the model by defining the class.

18. The universal approximation theorem shows that any function can be approximated as closely as needed using 
just one nonlinearity. So why do we normally use more?

An artificial neural network with two layers and a nonlinear activation function can approximate any function but 
there are performance benefits for using more layers. It turns out that smaller matrices with more layers perform 
better than large matrices with fewer layers. It also means the model will train faster, use fewer parameters, and 
take up less memory.

19. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?

The images are resized from the largest size, the original size, to a large size on the CPU to produce higher 
quality images for training the model to create a more accurate model.

* It resizes the images to a large size to provide extra space to reduce the empty space and lost detail that 
occurs from the data augmentations.
* It resizes the images to the same size and square shape to allow the GPU to perform the data augmentations 
which is much faster than the CPU.
* It applies the data augmentations as one interpolation instead of multiple interpolations to reduce the 
image quality that’s lost from interpolation.

20. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.

Ok.

21. What are the two ways in which data is most commonly provided, for most deep learning datasets?

The majority of the datasets are organized into the following two formats:  
* Individual Files: The image, video, audio, or text files that represent the data items in the dataset. 
These files are organized into subdirectories with file names that represent information about the data items.
* Data Table: The comma-separated values (CSV) file that contains the data items in the dataset. 
This file contains the data items directly in the rows and columns which may include file paths to individual files.

22. Look up the documentation for L and try using a few of the new methods is that it adds.

* Concat: A method that combines the items in the list objects in the L object into an L object. 
It also returns the combined items in an L object.
* Shuffle: A method that reorganizes the order of the items in the L object. 
It also returns the reordered items in an L object.
* Zip: A method that pairs the items at each index of the list objects in the L object. 
It also returns the item pairs in tuple objects in an L object.

`L([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).concat()` -> `[1, 2, 3, 4, 5, 6, 7, 8, 9]`

23. Look up the documentation for the Python pathlib module and try using a few methods of the Path class.

* Mk_Write: A method that writes the specified data to the specified file in the specified path in the Path object. 
It creates any missing directories in the path in the Path object. 
It also passes the string literal to the data parameter in the method to set the data to write.
* ReadLines: A method that reads the data in the specified file in the specified path in the Path object. 
It also returns the data in a list object. 
* Ls: A method that iterates through the files and subdirectories in the specified path in the Path object. 
It also returns the paths to the files and subdirectories in Path objects in an L object.

`Path("./new_directory/").ls()`

24. Give two examples of ways that image transformations can degrade the quality of the data.

Interpolation: Image transformations uses a geometric function to produce a modified view of an image. 
It usually involves adding pixel values to the image that are estimated using the existing pixel values that 
surround the new pixel value. It also negatively affects the quality of the image based on the type, intensity, 
and quantity of transformations.

* Affine: Translates, scales, and rotates in one operation.
* Rotate: Rotates an image at a given point.
* Scale Enlarges or shrinks an image.
* Shear: Stretches an image vertically and horizontally at diagonal corners.
* Translate: Moves an image up, down, left, or right.
* Warp: Stretches an image at defined points.

Performing interpolations creates an image using pixel values that are estimated using estimated pixel values 
which loses quality each time.

Rotating an image 45 degrees creates empty space in the corners.

25. What method does fastai provide to view the data in a DataLoaders?

Show_Batch: A method that displays a specified number of images from a batch in the dataset in the DataLoaders object. 
It can set the max_n, nrows, and ncols parameters to an integer value to display a specific number of images. 
It can also set the unique parameter to the True boolean to display the same image multiple times with all the 
different data augmentations.

26. What method does fastai provide to help you debug a DataBlock?

Summary: A method that creates a batch using the items in the specified path to display information at each step in 
the transformation process. It must set the source parameter to a string value or Path object to specify the path. 
It can set the bs parameter to an integer value to specify the number of items to include in the batch. It can also 
set the show_batch parameter to the True boolean to display the number of items in the batch.

27. Should you hold off on training a model until you have thoroughly cleaned your data?

The Fastai approach is to train the model right after the DataBlock and DataLoaders objects are ready. 
This is because the trained model is used with the cleaning tools from the second chapter of the textbook to help 
clean the data. It also retrains the model after the data has been cleaned.

* Plot_Confusion_Matrix: A method that displays a confusion matrix to visualize the number of correct and incorrect 
predictions in each class.
* Plot_Top_Losses: A method that displays the images with the highest loss value. It also displays the associated 
predicted, label, loss, and probability values.
* ImageClassifierCleaner: A class that displays a widget to iterate through the images with the highest loss value 
to manually relabel or remove each image.

28. What are the two pieces that are combined into cross-entropy loss in PyTorch?

Cross Entropy Loss: A function that calculates the loss for multi-class classification tasks. It combines the 
LogSoftmax function and the NLLLoss function. It passes the input tensor to the LogSoftmax function to calculate 
the logarithm of the softmax. It passes the logarithm of the softmax that's returned by the LogSoftmax function to 
the NLLLoss function to calculate the loss. It also returns the loss as a float value in the output tensor.

* Log of the Softmax (LogSoftmax): A function that calculates the logarithm of the softmax. 
It combines the Log function and the Softmax function. 
It passes the input tensor to the Softmax function to convert the predicted values to probability values. 
It passes the probability values that are returned by the Softmax function to the Log function to calculate the 
logarithm of the probability values. 
It also returns the logarithms as float values in the output tensor. 
* Negative Log Likelihood Loss (NLLLoss): A function that calculates the loss using the logarithm of the softmax. 
It uses the indexing syntax to access the loss values in the input tensor using the label values in the input tensor. 
It accesses the loss values in all the row indexes but only for the column index that represents the correct label. 
It calculates the mean of the loss values that are returned by the indexing operation. 
It also returns the mean as a float value in the output tensor.

29. What are the two properties of activations that softmax ensures? Why is this important?

Softmax: A function that's commonly used as the final layer in the deep learning model to convert the predicted 
values into probability values. It calculates the probability values by dividing the exponential of the predicted 
values by the sum of the exponential of the predicted values. It also returns the probability values as float values 
in the output tensor.

The exp method amplifies the small differences between the predicted values. It makes the larger predicted values 
much larger than the rest. It also makes the largest predicted value much larger than the larger predicted values.

`torch.exp(x) / torch.exp(x).sum(dim = 1, keepdim = True)`

* It ensures the probability values range between the 0.0 and 1.0 float.
* It ensures the probability values add up to equal the 1.0 float.

This is important because the Softmax function is built to produce the best predicted value which is the 
probability value that's closest to the 1.0 float. It also means the Softmax function always predicts a label value 
which is good for image classification when each image has a definite label.

30. When might you want your activations to not have these two properties?

Multi-Label Classification: An image classification task where each image has zero or more instances of two or more 
labels. It uses the sigmoid function in the final layer of the deep learning model to convert the predicted values 
in the input tensor into probability values that range between the 0.0 and 1.0 float. It also treats each label as 
its own binary classification task that’s calculated separately from the other labels.

31. Calculate the exp and softmax columns of <<bear_softmax>> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).

Ok.

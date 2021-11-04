# Lesson 3

## Recap of Lesson 2 + What's next

Finish production stuff and learn about what goes on when we train a neural network. 

## Resizing Images with DataBlock

When we use Resize, we resize all images by squishing them to 128px. Images have to be squares and the same size.

You can then use `bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))`.
* Center: Grab the center of the image. Default.
* Squish: Squish the width to the specified size.
* Pad: Add empty bars to the top or bottom to achieve the desired aspect ratio.
* RandomResizedCrop: Each time, grab a different part of the image, and zoom into it. The most common approach. 

## Data Augmentation and item_tfms vs batch_tfms

Data Augmentation is the idea of doing something so that the model sees an image differently each time.

One of the best ways is to use `aug_transforms`.
* `batch_tfms`: Batch Transforms. Happen in groups and in the GPU.
* `item_tfms`: Item Transforms. Transforms one image at a time.

## Training your model, and using it to clean your data

You can use `ImageClassifierCleaner` to select, delete, or relabel image for training or validating.

## Turning your model into an online application

You can export stuff into a pkl file and then load it into a learner and call predict.

The tensor returned by predict is the most important value. You can get the mapping via `learn_inf.dls.vocab`.

The course uses IPython widgets (ipywidgets) and Voilà to publish apps.
* A FileUpload widget allows you to upload images.
* An Output widget allows you to set up a placeholder.
* A Label widget allows you to add text.
* A Button widget allows you to add a button with a click event handler.

A VBox is a vertical box to create your GUI.

Voilà takes a notebook and only displays the widgets. This lets you run it in a mobile app.

Binder is a tool to paste the path to your notebook on GitHub and generates a URL that anyone can use. 

Things are deployed using a CPU, not a GPU. This is fine since you're not processing many images. CPU is cheaper.

## Deploying to a mobile phone

Deploy to a server, then have the mobile app communicate to the server. Trying to run PyTorch on the phone is 
difficult.

## How to avoid disaster

The world is full of biased data. Bing Image Search returns a bunch of white women for "healthy skin". This becomes a
"young white woman touching her face" detector. Be careful. Gather data that reflects the real world.

"Building Machine Learning Powered Applications" book is good at pointing this out.

Out of Domain Data: Data that our model sees in production which is very different to what it saw during training.

Ways to address out of domain data:
* Having a diverse team.
* Writing the attributes of your data sets and publishing its limitations.

Domain Shift: When the type of data that our model sees changes over time. 

Recommendation:
1. Manual process
   1. Run model in parallel
   2. Humans check predictions
2. Limited scope deployment
   1. Careful human supervision
   2. Time or geography limited
3. Gradual expansion
   1. Good reporting systems needed
   2. Consider what could go wrong

## Unforeseen consequences and feedback loops

Feedback loops are challenging for real world deployments. A minor issue can explode into a big issue.
Ex: Predictive policing.

A good way to go about it to ask yourself, "What would happen if a ML system went really well? What would extreme
results look like?"

You want to add points where humans can observe and change things. Don't want to have it isolated from product and 
engineers.

## End of Chapter 2 Recap + Blogging

Writing is important. Things get more complicated. Writing down what you learned helps solidify learning.

## Starting MNIST from scratch

We're going to try to recognize hand-written digits from scratch. 

First thing is to build a classifier that can recognize if a digit is a 3 or a 7.

## untar_data and path explained

`untar_data` = fastai function used to take a url, download data, uncompress it, and return path of where the data is.

`path` = A pathlib PosixPath object that helps us navigate path to files.

## Exploring at the MNIST data

The path to `/path/train/3/<IMG>.png` has all the 3's.  
The path to `/path/train/7/<IMG>.png` has all the 7's.  

PIL is the Python Image Library.

## NumPy Array vs PyTorch Tensor

NumPy Array converts the image to an array of numbers. You can also do this with a Tensor in PyTorch.
PyTorch Tensors have the advantage of GPU computation.

## Creating a simple baseline model

1. Try Pixel Similarity. Find the average pixel value for every pixel of threes and same for seven. This will give
us two groups of averages, and when we pass a digit, we can see if it's closer to the "ideal" 3 or 7.
   * Make a list of all the sevens and threes and turn them into tensors.
   * show_image can display a tensor as an image
   * We can stack all the threes and sevens to gauge the pixel intensity using torch.stack.
   * Rank is the number of dimensions in a tensor (i.e. its length). 
   * Shape is the size of each dimension of a tensor.
   * You have to convert the intensities in the tensors to be positive so we can calculate distance.
   * You could use absolute value (mean absolute difference or L1 norm), or 
   * the mean of the square of differences and then take the square root (root mean squared error RMSE or L2 norm).
   * When we try them both, they both correctly determine that the image is a 3.

## Working with arrays and tensors

Consider the following data: `data = [[1, 2, 3], [4, 5, 6]]`

NumPy arrays and PyTorch tensors look the same. You can index into a row like `tns[1]`,
you can index into a column like `tns[:,1]`. 

`:` means all of the first dimension. 

You can combine and use like Python slice syntax `[start:end]`, with `end` being excluded.
`tns[1, 1:3]` = `tensor([5, 6])`

You can add: `tns+1` will add 1 to each value.

Other operations: 
* `tns.type()` -> torch.LongTensor.
* `tns*1.5` -> Converts tensor from int to float.

## Computing metrics with Broadcasting
## Stochastic Gradient Descent (SGD)
## End-to-end Gradient Descent example
## MNIST loss function
## Lesson 3 review 

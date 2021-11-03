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

The course uses IPython widgets (ipywidgets) and Voila to publish apps.
* A FileUpload widget allows you to upload images.
* An Output widget allows you to set up a placeholder.
* A Label widget allows you to add text.
* A Button widget allows you to add a button with a click event handler.

A VBox is a vertical box to create your GUI.

Voila takes a notebook and only displays the widgets. This lets you run it in a mobile app.

Binder is a tool to paste the path to your notebook on Github and generates a URL that anyone can use. 

Things are deployed using a CPU, not a GPU. This is fine since you're not processing many images. CPU is cheaper.

## Deploying to a mobile phone
## How to avoid disaster
## Unforeseen consequences and feedback loops
## End of Chapter 2 Recap + Blogging
## Starting MNIST from scratch
## untar_data and path explained
## Exploring at the MNIST data
## NumPy Array vs PyTorch Tensor
## Creating a simple baseline model
## Working with arrays and tensors
## Computing metrics with Broadcasting
## Stochastic Gradient Descent (SGD)
## End-to-end Gradient Descent example
## MNIST loss function
## Lesson 3 review 

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

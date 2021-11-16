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
## One hot encoding
## Regression
## Embedding
## Collaborative filtering from scratch
## Regularisation (Data augmentation for regression)

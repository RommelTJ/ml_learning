# Questionnaire Answers

1. Do you need these for deep learning?
* Lots of math T / F
* Lots of data T / F
* Lots of expensive computers T / F
* A PhD T / F

No.

2. Name five areas where deep learning is now the best in the world.

* NLP
* Computer Vision
* Medicine
* Biology
* Image generation

3. What was the name of the first device that was based on the principle of the artificial neuron?

Mark I Perceptron by Frank Rosenblatt

5. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?

* A set of processing units
* A state of activation
* An output function for each unit
* A pattern of connectivity among units
* A propagation rule for propagating patterns of activities through the network of connectivities
* An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce a new level of activation for the unit
* A learning rule whereby patterns of connectivity are modified by experience
* An environment within which the system must operate

5. What were the two theoretical misunderstandings that held back the field of neural networks?

People pointed out that AI models couldn't learn simple functions like XOR, even though the
limitations could be addressed.

In 1986, Parallel Distributed Processing (PDP) was released that described an approach to
deep learning.

6. What is a GPU?

Graphics Processing Unit.

7. Open a notebook and execute a cell containing: 1+1. What happens?

It executes. Code is run by Python and output is shown like a REPL.

8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.

Done.

9. Complete the Jupyter Notebook online appendix.

Done.

10. Why is it hard to use a traditional computer program to recognize images in a photo?

Because it's hard to define the set of algorithmic rules to do this.

11. What did Samuel mean by "weight assignment"?

The current values of the model parameters.

12. What term do we normally use in deep learning for what Samuel called "weights"?

Parameters.

13. Draw a picture that summarizes Samuel's view of a machine learning model.

`inputs + weights -> model -> results -> performance`

14. Why is it hard to understand why a deep learning model makes a particular prediction?

Because of their "deep" nature. Deep neural networks have hundreds/thousands of layers, and it 
is hard to understand which factors are most important to determine the final output. The field
is called "interpretability of deep learning models".

15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?

The Universal Approximation Theorem.

16. What do you need in order to train a model?

An architecture, data, and labels. You also need a loss function and a way to update the parameters
to the model to improve its performance.

17. How could a feedback loop impact the rollout of a predictive policing model?

It could bias the results. 

18. Do we always have to use 224×224-pixel images with the cat recognition model?

No.

19. What is the difference between classification and regression?

Classification is focused on predicting a class or category.
Regression is focused on predicting a numeric quantity.

20. What is a validation set? What is a test set? Why do we need them?

Validation Set: A portion of the dataset not used to train the model. Used to evaluate the model to prevent overfitting.

Test Set: A portion of the dataset used for final evaluation of a model. Used to evaluate model performance.

These sets of data are necessary to ensure that the model generalizes to unseen data.

21. What will fastai do if you don't provide a validation set?

It will take 20% of the data at random and assign it as the validation set.

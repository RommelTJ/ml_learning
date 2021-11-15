# Questionnaire

1. Why can't we use torch.where to create a loss function for datasets where our label can have more than two categories?
2. What is the value of log(-2)? Why?
3. What are two good rules of thumb for picking a learning rate from the learning rate finder?
4. What two steps does the fine_tune method do?
5. In Jupyter Notebook, how do you get the source code for a method or function?
6. What are discriminative learning rates?
7. How is a Python slice object interpreted when passed as a learning rate to fastai?
8. Why is early stopping a poor choice when using 1cycle training?
9. What is the difference between resnet50 and resnet101?
10. What does to_fp16 do?
11. How could multi-label classification improve the usability of the bear classifier?
12. How do we encode the dependent variable in a multi-label classification problem?
13. How do you access the rows and columns of a DataFrame as if it was a matrix?
14. How do you get a column by name from a DataFrame?
15. What is the difference between a Dataset and DataLoader?
16. What does a Datasets object normally contain?
17. What does a DataLoaders object normally contain?
18. What does lambda do in Python?
19. What are the methods to customize how the independent and dependent variables are created with the data block API?
20. Why is softmax not an appropriate output activation function when using a one hot encoded target?
21. Why is nll_loss not an appropriate loss function when using a one-hot-encoded target?
22. What is the difference between nn.BCELoss and nn.BCEWithLogitsLoss?
23. Why can't we use regular accuracy in a multi-label problem?
24. When is it okay to tune a hyperparameter on the validation set?
25. How is y_range implemented in fastai? (See if you can implement it yourself and test it without peeking!)
26. What is a regression problem? What loss function should you use for such a problem?
27. What do you need to do to make sure the fastai library applies the same data augmentation to your inputs images and your target point coordinates?
28. What problem does collaborative filtering solve?
29. How does it solve it?
30. Why might a collaborative filtering predictive model fail to be a very useful recommendation system?
31. What does a crosstab representation of collaborative filtering data look like?
32. Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching!).
33. What is a latent factor? Why is it "latent"?
34. What is a dot product? Calculate a dot product manually using pure Python with lists.
35. What does pandas.DataFrame.merge do?
36. What is an embedding matrix?
37. What is the relationship between an embedding and a matrix of one-hot-encoded vectors?
38. Why do we need Embedding if we could use one-hot-encoded vectors for the same thing?
39. What does an embedding contain before we start training (assuming we're not using a pretained model)?
40. Create a class (without peeking, if possible!) and use it.
41. What does x[:,0] return?
42. Rewrite the DotProduct class (without peeking, if possible!) and train a model with it.
43. What is a good loss function to use for MovieLens? Why?
44. What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?
45. What is the use of bias in a dot product model?

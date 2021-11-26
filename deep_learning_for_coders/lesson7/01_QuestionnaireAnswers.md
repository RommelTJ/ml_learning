# Questionnaire Answers

1. What is another name for weight decay?

L2 regularization

2. Write the equation for weight decay (without peeking!).

`loss_with_wd = loss + wd * (parameters**2).sum()`

3. Write the equation for the gradient of weight decay. Why does it help reduce weights?

We add to the gradients 2*wd*parameters. This helps create more shallow, less bumpy/sharp surfaces that generalize 
better and prevents overfitting.

4. Why does reducing weights lead to better generalization?

This will result is more shallow, less sharp surfaces. If sharp surfaces are allowed, it can very easily overfit, and 
now this is prevented.

5. What does argsort do in PyTorch?

This just gets the indices in the order that the original PyTorch Tensor is sorted.

6. Does sorting the movie biases give the same result as averaging overall movie ratings by movie? Why/why not?

No it means much more than that. It takes into account the genres or actors or other factors. For example, movies 
with low bias means even if you like these types of movies you may not like this movie (and vice versa for movies 
with high bias).

7. How do you print the names and details of the layers in a model?

Just by typing learn.model.

8. What is the "bootstrapping problem" in collaborative filtering?

That the model / system cannot make any recommendations or draw any inferences for users or items about which it has 
not yet gathered sufficient information. It’s also called the cold start problem.

9. How could you deal with the bootstrapping problem for new users? For new movies?

You could solve this by coming up with an average embedding for a user or movie. Or select a particular user/movie to 
represent the average user/movie. Additionally, you could come up with some questions that could help initialize the 
embedding vectors for new users and movies.

10. How can feedback loops impact collaborative filtering systems?

The recommendations may suffer from representation bias where a small number of people influence the system 
heavily. E.g.: Highly enthusiastic anime fans who rate movies much more frequently than others may cause the system 
to recommend anime more often than expected (incl. to non-anime fans).

11. When using a neural network in collaborative filtering, why can we have different numbers of factors for movies 
    and users?

In this case, we are not taking the dot product but instead concatenating the embedding matrices, so the number of 
factors can be different.

12. Why is there an nn.Sequential in the CollabNN model?

This allows us to couple multiple nn.Module layers together to be used. In this case, the two linear layers are 
coupled together and the embeddings can be directly passed into the linear layers.

13. What kind of model should we use if we want to add metadata about users and items, or information such as date and 
    time, to a collaborative filtering model?

We should use a tabular model.

14. What is a continuous variable?

This refers to numerical variables that have had a wide range of "continuous" values (ex: age).

15. What is a categorical variable?

This refers to variables that can take on discrete levels that correspond to different categories.

16. Provide two of the words that are used for the possible values of a categorical variable.

Levels or categories

17. What is a "dense layer"?

Equivalent to what we call linear layers.

18. How do entity embeddings reduce memory usage and speed up neural networks?

Especially for large datasets, representing the data as one-hot encoded vectors can be very inefficient (and also 
sparse). On the other hand, using entity embeddings allows the data to have a much more memory-efficient (dense) 
representation of the data. This will also lead to speed-ups for the model.

19. What kinds of datasets are entity embeddings especially useful for?

It is especially useful for datasets with features that have high levels of cardinality (the features have lots of 
possible categories). Other methods often overfit to data like this.

20. What are the two main families of machine learning algorithms?

* Ensemble of decision trees are best for structured (tabular data)
* Multilayered neural networks are best for unstructured data (audio, vision, text, etc.)

21. Why do some categorical columns need a special ordering in their classes? How do you do this in Pandas?

Ordinal categories may inherently have some order and by using set_categories with the argument ordered=True and 
passing in the ordered list, this information represented in the pandas DataFrame.

22. Summarize what a decision tree algorithm does.

The basic idea of what a decision tree algorithm does is to determine how to group the data based on "questions" 
that we ask about the data. That is, we keep splitting the data based on the levels or values of the features and 
generate predictions based on the average target value of the data points in that group. Here is the algorithm:
* Loop through each column of the dataset in turn
* For each column, loop through each possible level of that column in turn
* Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is 
a categorical variable, based on whether they are equal to or not equal to that level of that categorical variable)
* Find the average sale price for each of those two groups, and see how close that is to the actual sale price of 
each of the items of equipment in that group. That is, treat this as a very simple “model” where our predictions are 
simply the average sale price of the item’s group.
* After looping through all of the columns and possible levels for each, pick the split point which gave the best 
predictions using our very simple model.
* We now have two different groups for our data, based on this selected split. Treat each of these as separate 
datasets, and find the best split for each, by going back to step one for each group.
* Continue this process recursively, and until you have reached some stopping criterion for each group — for instance, 
stop splitting a group further when it has only 20 items in it.

23. Why is a date different from a regular categorical or continuous variable, and how can you preprocess it to allow 
    it to be used in a model?

Some dates are different to others (ex: some are holidays, weekends, etc.) that cannot be described as just an ordinal 
variable. Instead, we can generate many different categorical features about the properties of the given date 
(ex: is it a weekday? is it the end of the month?, etc.).

24. Should you pick a random validation set in the bulldozer competition? If no, what kind of validation set should 
    you pick?

No, the validation set should be as similar to the test set as possible. In this case, the test set contains data 
from later data, so we should split the data by the dates and include the later dates in the validation set.

25. What is pickle and what is it useful for?

Allows you so save nearly any Python object as a file.

26. How are mse, samples, and values calculated in the decision tree drawn in this chapter?

By traversing the tree based on answering questions about the data, we reach the nodes that tell us the average value 
of the data in that group, the mse, and the number of samples in that group.

27. How do we deal with outliers, before building a decision tree?

Finding out of domain data (Outliers)  
Sometimes it is hard to even know whether your test set is distributed in the same way as your training data or, if 
it is different, then what columns reflect that difference. There’s actually a nice easy way to figure this out, 
which is to use a random forest!  
But in this case we don't use a random forest to predict our actual dependent variable. Instead, we try to predict 
whether a row is in the validation set, or the training set.

28. How do we handle categorical variables in a decision tree?

We convert the categorical variables to integers, where the integers correspond to the discrete levels of the 
categorical variable. Apart from that, there is nothing special that needs to be done to get it to work with decision 
trees (unlike neural networks, where we use embedding layers).

29. What is bagging?

Train multiple models on random subsets of the data, and use the ensemble of models for prediction.

30. What is the difference between max_samples and max_features when creating a random forest?

When training random forests, we train multiple decision trees on random subsets of the data. max_samples defines how 
many samples, or rows of the tabular dataset, we use for each decision tree. max_features defines how many features, 
or columns of the tabular dataset, we use for each decision tree.

31. If you increase n_estimators to a very high value, can that lead to overfitting? Why or why not?

A higher n_estimators mean more decision trees are being used. However, since the trees are independent of each other, 
using higher n_estimators does not lead to overfitting.

32. In the section "Creating a Random Forest", just after <<max_features>>, why did preds.mean(0) give the same 
    result as our random forest?

Done.

33. What is "out-of-bag-error"?

Only use the models not trained on the row of data when going through the data and evaluating the dataset. 
No validation set is needed.

34. Make a list of reasons why a model's validation set error might be worse than the OOB error. How could you test 
    your hypotheses?

The major reason could be because the model does not generalize well. Related to this is the possibility that the 
validation data has a slightly different distribution than the data the model was trained on.

35. Explain why random forests are well suited to answering each of the following question:
* How confident are we in our predictions using a particular row of data?
* For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
* Which columns are the strongest predictors?
* How do predictions vary as we vary these columns?

Look at standard deviation between the estimators.

Using the treeinterpreter package to check how the prediction changes as it goes through the tree, adding up the 
contributions from each split/feature. Use waterfall plot to visualize.

Look at feature importance.

Look at partial dependence plots.

36. What's the purpose of removing unimportant variables?

Sometimes, it is better to have a more interpretable model with less features, so removing unimportant variables 
helps in that regard.

37. What's a good type of plot for showing tree interpreter results?

Waterfall plot.

38. What is the "extrapolation problem"?

Hard for a model to extrapolate to data that's outside the domain of the training data. This is particularly important 
for random forests. On the other hand, neural networks have underlying Linear layers so it could potentially 
generalize better.

39. How can you tell if your test or validation set is distributed in a different way than your training set?

We can do so by training a model to classify if the data is training or validation data. If the data is of different 
distributions (out-of-domain data), then the model can properly classify between the two datasets.

40. Why do we make saleElapsed a continuous variable, even although it has less than 9,000 distinct values?

This is a variable that changes over time, and since we want our model to extrapolate for future results, we make 
this a continuous variable.

41. What is "boosting"?

We train a model that underfits the dataset, and train subsequent models that predicts the error of the original model. 
We then add the predictions of all the models to get the final prediction.

42. How could we use embeddings with a random forest? Would we expect this to help?

Entity embeddings contains richer representations of the categorical features and definitely can improve the 
performance of other models like random forests. Instead of passing in the raw categorical columns, the entity 
embeddings can be passed into the random forest model.

43. Why might we not always use a neural net for tabular modeling?

We might not use them because they are the hardest to train and longest to train, and less well-understood. Instead, 
random forests should be the first choice/baseline, and neural networks could be tried to improve these results or add 
to an ensemble.

# Questionnaire

1. What is another name for weight decay?
2. Write the equation for weight decay (without peeking!).
3. Write the equation for the gradient of weight decay. Why does it help reduce weights?
4. Why does reducing weights lead to better generalization?
5. What does argsort do in PyTorch?
6. Does sorting the movie biases give the same result as averaging overall movie ratings by movie? Why/why not?
7. How do you print the names and details of the layers in a model?
8. What is the "bootstrapping problem" in collaborative filtering?
9. How could you deal with the bootstrapping problem for new users? For new movies?
10. How can feedback loops impact collaborative filtering systems?
11. When using a neural network in collaborative filtering, why can we have different numbers of factors for movies and users?
12. Why is there an nn.Sequential in the CollabNN model?
13. What kind of model should we use if we want to add metadata about users and items, or information such as date and time, to a collaborative filtering model?
14. What is a continuous variable?
15. What is a categorical variable?
16. Provide two of the words that are used for the possible values of a categorical variable.
17. What is a "dense layer"?
18. How do entity embeddings reduce memory usage and speed up neural networks?
19. What kinds of datasets are entity embeddings especially useful for?
20. What are the two main families of machine learning algorithms?
21. Why do some categorical columns need a special ordering in their classes? How do you do this in Pandas?
22. Summarize what a decision tree algorithm does.
23. Why is a date different from a regular categorical or continuous variable, and how can you preprocess it to allow it to be used in a model?
24. Should you pick a random validation set in the bulldozer competition? If no, what kind of validation set should you pick?
25. What is pickle and what is it useful for?
26. How are mse, samples, and values calculated in the decision tree drawn in this chapter?
27. How do we deal with outliers, before building a decision tree?
28. How do we handle categorical variables in a decision tree?
29. What is bagging?
30. What is the difference between max_samples and max_features when creating a random forest?
31. If you increase n_estimators to a very high value, can that lead to overfitting? Why or why not?
32. In the section "Creating a Random Forest", just after <<max_features>>, why did preds.mean(0) give the same result as our random forest?
33. What is "out-of-bag-error"?
34. Make a list of reasons why a model's validation set error might be worse than the OOB error. How could you test your hypotheses?
35. Explain why random forests are well suited to answering each of the following question:
  * How confident are we in our predictions using a particular row of data?
  * For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
  * Which columns are the strongest predictors?
  * How do predictions vary as we vary these columns?
36. What's the purpose of removing unimportant variables?
37. What's a good type of plot for showing tree interpreter results?
38. What is the "extrapolation problem"?
39. How can you tell if your test or validation set is distributed in a different way than your training set?
40. Why do we make saleElapsed a continuous variable, even although it has less than 9,000 distinct values?
41. What is "boosting"?
42. How could we use embeddings with a random forest? Would we expect this to help?
43. Why might we not always use a neural net for tabular modeling?


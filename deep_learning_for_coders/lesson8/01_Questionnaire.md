# Questionnaire

1. What is "self-supervised learning"?
2. What is a "language model"?
3. Why is a language model considered self-supervised?
4. What are self-supervised models usually used for?
5. Why do we fine-tune language models?
6. What are the three steps to create a state-of-the-art text classifier?
7. How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?
8. What are the three steps to prepare your data for a language model?
9. What is "tokenization"? Why do we need it?
10. Name three different approaches to tokenization.
11. What is xxbos?
12. List four rules that fastai applies to text during tokenization.
13. Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?
14. What is "numericalization"?
15. Why might there be words that are replaced with the "unknown word" token?
16. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)
17. Why do we need padding for text classification? Why don't we need it for language modeling?
18. What does an embedding matrix for NLP contain? What is its shape?
19. What is "perplexity"?
20. Why do we have to pass the vocabulary of the language model to the classifier data block?
21. What is "gradual unfreezing"?
22. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?
23. If the dataset for your project is so big and complicated that working with it takes a significant amount of time, what should you do?
24. Why do we concatenate the documents in our dataset before creating a language model?
25. To use a standard fully connected network to predict the fourth word given the previous three words, what two tweaks do we need to make to ou model?
26. How can we share a weight matrix across multiple layers in PyTorch?
27. Write a module that predicts the third word given the previous two words of a sentence, without peeking.
28. What is a recurrent neural network?
29. What is "hidden state"?
30. What is the equivalent of hidden state in LMModel1?
31. To maintain the state in an RNN, why is it important to pass the text to the model in order?
32. What is an "unrolled" representation of an RNN?
33. Why can maintaining the hidden state in an RNN lead to memory and performance problems? How do we fix this problem?
34. What is "BPTT"?
35. Write code to print out the first few batches of the validation set, including converting the token IDs back into English strings, as we showed for batches of IMDb data in <<chapter_nlp>>.
36. What does the ModelResetter callback do? Why do we need it?
37. What are the downsides of predicting just one output word for each three input words?
38. Why do we need a custom loss function for LMModel4?
39. Why is the training of LMModel4 unstable?
40. In the unrolled representation, we can see that a recurrent neural network actually has many layers. So why do we need to stack RNNs to get better results?
41. Draw a representation of a stacked (multilayer) RNN.
42. Why should we get better results in an RNN if we call detach less often? Why might this not happen in practice with a simple RNN?
43. Why can a deep network result in very large or very small activations? Why does this matter?
44. In a computer's floating-point representation of numbers, which numbers are the most precise?
45. Why do vanishing gradients prevent training?
46. Why does it help to have two hidden states in the LSTM architecture? What is the purpose of each one?
47. What are these two states called in an LSTM?
48. What is tanh, and how is it related to sigmoid?
49. What is the purpose of this code in LSTMCell: h = torch.stack([h, input], dim=1)
50. What does chunk do in PyTorch?
51. Study the refactored version of LSTMCell carefully to ensure you understand how and why it does the same thing as the non-refactored version.
52. Why can we use a higher learning rate for LMModel6?
53. What are the three regularization techniques used in an AWD-LSTM model?
54. What is "dropout"?
55. Why do we scale the weights with dropout? Is this applied during training, inference, or both?
56. What is the purpose of this line from Dropout: if not self.training: return x
57. Experiment with bernoulli_ to understand how it works.
58. How do you set your model in training mode in PyTorch? In evaluation mode?
59. Write the equation for activation regularization (in math or code, as you prefer). How is it different from weight decay?
60. Write the equation for temporal activation regularization (in math or code, as you prefer). Why wouldn't we use this for computer vision problems?
61. What is "weight tying" in a language model?

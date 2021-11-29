# Questionnaire Answers

1. What is "self-supervised learning"?

Training a model without the use of labels. An example is a language model.

2. What is a "language model"?

A language model is a self-supervised model that tries to predict the next word of a given passage of text.

3. Why is a language model considered self-supervised?

There are no labels (ex: sentiment) provided during training. Instead, the model learns to predict the next word by 
reading lots of provided text with no labels.

4. What are self-supervised models usually used for?

Sometimes, they are used by themselves. For example, a language model can be used for autocomplete algorithms! But 
often, they are used as a pre-trained model for transfer learning.

5. Why do we fine-tune language models?

We can fine-tune the language model on the corpus of the desired downstream task, since the original pre-trained 
language model was trained on a corpus that is slightly different than the one for the current task.

6. What are the three steps to create a state-of-the-art text classifier?

* Train a language model on a large corpus of text (already done for ULM-FiT by Sebastian Ruder and Jeremy!).
* Fine-tune the language model on text classification dataset.
* Fine-tune the language model as a text classifier instead.

7. How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?

By learning how to predict the next word of a movie review, the model better understands the language style and 
structure of the text classification dataset and can, therefore, perform better when fine-tuned as a classifier.

8. What are the three steps to prepare your data for a language model?

* Tokenization
* Numericalization
* Language model DataLoader

9. What is "tokenization"? Why do we need it?

Tokenization is the process of converting text into a list of words. It is not as simple as splitting on the spaces. 
Therefore, we need a tokenizer that deals with complicated cases like punctuation, hyphenated words, etc.

10. Name three different approaches to tokenization.

* Word-based tokenization
* Subword-based tokenization
* Character-based tokenization

11. What is xxbos?

This is a special token added by fastai that indicated the beginning of the text.

12. List four rules that fastai applies to text during tokenization.

Here are all the rules:
* `fix_html` replace special HTML characters by a readable version (IMDb reviews have quite a few of them for instance).
* `replace_rep` replace any character repeated three times or more by a special token for repetition (xxrep), the 
number of times it’s repeated, then the character
* `replace_wrep` replace any word repeated three times or more by a special token for word repetition (xxwrep), the 
number of times it’s repeated, then the word
* `spec_add_spaces` add spaces around / and #
* `rm_useless_spaces` remove all repetitions of the space character
* `replace_all_caps` lowercase a word written in all caps and adds a special token for all caps (xxcap) in front of it
* `replace_maj` lowercase a capitalized word and adds a special token for capitalized (xxmaj) in front of it
* `lowercase` lowercase all text and adds a special token at the beginning (xxbos) and/or the end (xxeos)

13. Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?

We can expect that repeated characters could have special or different meaning than just a single character. By 
replacing them with a special token showing the number of repetitions, the model’s embedding matrix can encode 
information about general concepts such as repeated characters rather than requiring a separate token for every 
number of repetitions of every character.

14. What is "numericalization"?

This refers to the mapping of the tokens to integers to be passed into the model.

15. Why might there be words that are replaced with the "unknown word" token?

If all the words in the dataset have a token associated with them, then the embedding matrix will be very large, 
increase memory usage, and slow down training. Therefore, only words with more than min_freq occurrence are assigned 
a token and finally a number, while others are replaced with the "unknown word" token.

16. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens 
for the dataset. What does the second row of that tensor contain? What does the first row of the second batch 
contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)

a. The dataset is split into 64 mini-streams (batch size)
b. Each batch has 64 rows (batch size) and 64 columns (sequence length)
c. The first row of the first batch contains the beginning of the first mini-stream (tokens 1-64)
d. The second row of the first batch contains the beginning of the second mini-stream
e. The first row of the second batch contains the second chunk of the first mini-stream (tokens 65-128)

17. Why do we need padding for text classification? Why don't we need it for language modeling?

Since the documents have variable sizes, padding is needed to collate the batch. Other approaches. like cropping or 
squishing, either to negatively affect training or do not make sense in this context. Therefore, padding is used. It 
is not required for language modeling since the documents are all concatenated.

18. What does an embedding matrix for NLP contain? What is its shape?

It contains vector representations of all tokens in the vocabulary. The embedding matrix has the size 
(vocab_size x embedding_size), where vocab_size is the length of the vocabulary, and embedding_size is an arbitrary 
number defining the number of latent factors of the tokens.

19. What is "perplexity"?

Perplexity is a commonly used metric in NLP for language models. It is the exponential of the loss.

20. Why do we have to pass the vocabulary of the language model to the classifier data block?

This is to ensure the same correspondence of tokens to index so the model can appropriately use the embeddings 
learned during LM fine-tuning.

21. What is "gradual unfreezing"?

This refers to unfreezing one layer at a time and fine-tuning the pretrained model.

22. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?

The classification models could be used to improve text generation algorithms (evading the classifier) so the text 
generation algorithms will always be ahead.

23. If the dataset for your project is so big and complicated that working with it takes a significant amount of time, 
what should you do?

Perhaps create the simplest possible dataset that allow for quick and easy prototyping. 
For example, Jeremy created a "human numbers" dataset.

24. Why do we concatenate the documents in our dataset before creating a language model?

To create a continuous stream of input/target words, to be able to split it up in batches of significant size.

25. To use a standard fully connected network to predict the fourth word given the previous three words, what two 
tweaks do we need to make to ou model?

* Use the same weight matrix for the three layers.
* Use the first word's embeddings as activations to pass to linear layer, add the second word’s embeddings to 
the first layer's output activations, and continues for rest of words.

26. How can we share a weight matrix across multiple layers in PyTorch?

Define one layer in the PyTorch model class, and use them multiple times in the forward method.

27. Write a module that predicts the third word given the previous two words of a sentence, without peeking.

```
class LMModel1(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        h = F.relu(self.h_h(self.i_h(x[:,0])))
        h = h + self.i_h(x[:,1])
        h = F.relu(self.h_h(h))
        h = h + self.i_h(x[:,2])
        h = F.relu(self.h_h(h))
        return self.h_o(h)
```

28. What is a recurrent neural network?

A refactoring of a multi-layer neural network as a loop.

29. What is "hidden state"?

The activations updated after each RNN step.

30. What is the equivalent of hidden state in LMModel1?

It is also defined as h in LMModel1.

31. To maintain the state in an RNN, why is it important to pass the text to the model in order?

Because state is maintained over all batches independent of sequence length, this is only useful if the text 
is passed in order.

32. What is an "unrolled" representation of an RNN?

A representation without loops, depicted as a standard multilayer network.

33. Why can maintaining the hidden state in an RNN lead to memory and performance problems? How do we fix this problem?

Since the hidden state is maintained through every single call of the model, when performing backpropagation with 
the model, it has to use the gradients from also all the past calls of the model. This can lead to high memory usage. 
So therefore after every call, the detach method is called to delete the gradient history of previous calls of the 
model.

34. What is "BPTT"?

Calculating backpropagation only for the given batch, and therefore only doing backprop for the defined sequence 
length of the batch.

35. Write code to print out the first few batches of the validation set, including converting the token IDs back into 
English strings, as we showed for batches of IMDb data in <<chapter_nlp>>.

Ok.

36. What does the ModelResetter callback do? Why do we need it?

It resets the hidden state of the model before every epoch and before every validation run.

37. What are the downsides of predicting just one output word for each three input words?

There are words in between that are not being predicted and that is extra information for training the model that is 
not being used. To solve this, we apply the output layer to every hidden state produced to predict three output 
words for the three input words (offset by one).

38. Why do we need a custom loss function for LMModel4?

CrossEntropyLoss expects flattened tensors.

39. Why is the training of LMModel4 unstable?

Because this network is effectively very deep and this can lead to very small or very large gradients that 
don't train well.

40. In the unrolled representation, we can see that a recurrent neural network actually has many layers. So why do we 
need to stack RNNs to get better results?

Because only one weight matrix is really being used. So multiple layers can improve this.

41. Draw a representation of a stacked (multilayer) RNN.

Ok.

42. Why should we get better results in an RNN if we call detach less often? Why might this not happen in practice with a simple RNN?

Ok.

43. Why can a deep network result in very large or very small activations? Why does this matter?

Numbers that are just slightly higher or lower than one can lead to the explosion or disappearance of numbers after 
repeated multiplications. In deep networks, we have repeated matrix multiplications, so this is a big problem.

44. In a computer's floating-point representation of numbers, which numbers are the most precise?

Small numbers, that are not too close to zero, however.

45. Why do vanishing gradients prevent training?

Gradients that are zero can't contribute to training because they don't change any weights.

46. Why does it help to have two hidden states in the LSTM architecture? What is the purpose of each one?

* One state remembers what happened earlier in the sentence.
* The other predicts the next token.

47. What are these two states called in an LSTM?

* Cell state (long short-term memory).
* Hidden state (predict next token).

48. What is tanh, and how is it related to sigmoid?

It's just a sigmoid function rescaled to the range of -1 to 1.

49. What is the purpose of this code in LSTMCell: h = torch.stack([h, input], dim=1)

This should actually be torch.cat([h, input], dim=1). It joins the hidden state and the new input.

50. What does chunk do in PyTorch?

Splits a tensor in equal sizes.

51. Study the refactored version of LSTMCell carefully to ensure you understand how and why it does the same thing as 
the non-refactored version.

Ok.

52. Why can we use a higher learning rate for LMModel6?

Because LSTM provides a partial solution to exploding/vanishing gradients.

53. What are the three regularization techniques used in an AWD-LSTM model?

* Dropout.
* Activation regularization.
* Temporal activation regularization.

54. What is "dropout"?

Deleting activations at random.

55. Why do we scale the weights with dropout? Is this applied during training, inference, or both?

* The scale changes if we sum up activations, it makes a difference if all activations are present or they are 
dropped with probability p. To correct the scale, a division by (1-p) is applied.
* In the implementation in the book, it is applied during training.
* It should be possible in both ways.

56. What is the purpose of this line from Dropout: if not self.training: return x

When not in training mode, don't apply dropout.

57. Experiment with bernoulli_ to understand how it works.

Ok.

58. How do you set your model in training mode in PyTorch? In evaluation mode?

Module.train(), Module.eval()

59. Write the equation for activation regularization (in math or code, as you prefer). How is it different from weight decay?

loss += alpha * activations.pow(2).mean()
It is different by not decreasing the weights but the activations

60. Write the equation for temporal activation regularization (in math or code, as you prefer). Why wouldn't we use 
this for computer vision problems?

This focuses on making the activations of consecutive tokens to be similar: 
loss += alpha * activations.pow(2).mean()

61. What is "weight tying" in a language model?

Weights of input-to-hidden layer is the same of weights of hidden-to-output layer is the same. This basically means 
we assume that the mapping from English words to activations.


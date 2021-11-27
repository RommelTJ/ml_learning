# Lesson 8

## Intro and NLP Review

Reminder that we say NLP in lesson 1. See notebook 1.

## Language models for NLP

The pre-trained language model was used to classify IMDB reviews (trained on Wikipedia).

Language models are used to predict the next word of a sentence.

Self-supervised learning: 
* Training a model using labels that are embedded in the independent variable, rather than requiring external labels.
* Ex: Training a model to predict the next word of text.

## Review of text classifier in Lesson 1

* Downloaded wikipedia pre-trained model
* fine-tuned the model
* Got 93% accuracy.

## Improving results with a domain-specific language model
## Language model from scratch
## Tokenization
## Word tokenizer
## Subword tokenizer
## Question: how can we determine if pre-trained model is suitable for downstream task?
## Numericalization
## Creating batches for language model
## LMDataLoader
## Creating language model data with DataBlock
## Fine-tuning a language model
## Saving and loading models
## Question: Can language models learn meaning?
## Text generation with language model
## Creating classification model
## Question: Is stemming and lemmatisation still used in practice?
## Handling different sequence lengths
## Fine-tuning classifier
## Questions
## Ethics and risks associated with text generation language models
## Language model from scratch
## Question: are there model interpretability tools for language models?
## Preparing the dataset for RNN: tokenization and numericalization
## Defining a simple language model
## Question: can you speed up fine-tuning the NLP model?
## Simple language model continued
## Recurrent neural networks (RNN)
## Improving our RNN
## Back propagation through time
## Ordered sequences and callbacks
## Creating more signal for model
## Multilayer RNN
## Exploding and vanishing gradients
## LSTM
## Questions
## Regularisation using Dropout
## AR and TAR regularisation
## Weight tying
## TextLearner
## Conclusion

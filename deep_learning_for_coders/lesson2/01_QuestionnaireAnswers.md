# Questionnaire Answers

1. Can we always use a random sample for a validation set? Why or why not?

No, time series forecasting doesn't work well with random samples. It needs the data to be split into different 
time periods where most the data is used as historical data for training and only the recent data is used as 
future data for validation. It also leads to overly optimistic results that generalize poorly because random 
sampling uses past and future data for the validation set.

2. What is overfitting? Provide an example.

Overfitting is an error in machine learning that occurs when the model performs well on the training data but doesn't 
generalize well on unseen data. It can be the result of overtraining, lack of validation, or improper validation, 
weight adjustments, and optimization attempts. It can also be the result of using training data that contains 
meaningless information.

4. What is a metric? How does it differ from "loss"?

The Evaluation Metric is a function that’s used in machine learning to test the performance of the model. It can 
measure the accuracy, precision, and recall of models with balanced datasets, or the Area Under the Curve and Receiver 
Operating Characteristics of models with imbalanced datasets. It can also return misleading results which is why 
multiple metrics are used.

The Loss Function is a function that's used in machine learning to evaluate how well an algorithm performs on the 
given data. It calculates the loss of each training iteration which measures the mathematical distance between the 
predicted value and the actual value. It also gets used to calculate the gradient during the training process which 
is needed to update the weights.

5. How can pretrained models help?

The Pretrained Model is a model that's used in machine learning to perform a specific task. It has been trained with 
a large dataset which contains the weights and biases that represent the features of the dataset. It can also be 
retrained to perform a similar task using transfer learning which produces a model with greater accuracy that requires 
less data, time, and resources.

6. What is the "head" of a model?

The Head is an analogy in machine learning that's used to represent the output layer of an artificial neural network. 
It interprets the model as a backbone plus a head where the backbone refers to the architecture of the model and the 
head refers to the final layer of the architecture. It also interprets transfer learning as replacing the head of a 
pretrained backbone.

7. What kinds of features do the early layers of a CNN find? How about the later layers?

The earlier layers of the convolutional neural network are used to detect the low-level features in an image. It 
learns to detect edges and colors in the first layer which becomes the building blocks to detect textures made 
from combinations of edges and colors in the second layer. It also continues to learn how to detect more sophisticated 
features with each additional layer.

The later layers of the convolutional neural network are used to detect the high-level features in an image. It 
learns to detect sophisticated patterns that resemble textures found in objects such as eyes, ears, and noses. It 
also eventually learns how to detect objects such as humans and animals which becomes the building blocks to detect 
the specific objects from the dataset.

8. Are image models only useful for photos?

No, non-image data can be classified with image models to achieve high accuracy as long as the data is transformed 
into an image. It can involve plotting data by placing similar features together and dissimilar features further 
apart which groups the neighboring features. It can also uncover hidden patterns and or relationships between sets 
of features in the data.

9. What is an "architecture"?

The Architecture is a template that’s used in machine learning to build neural networks. It defines the number, 
size, and type of layers that are used in the neural network which represents the mathematical function that's used 
to train the model. It can also represent any type of neural network for supervised, unsupervised, hybrid, or 
reinforcement learning.

10. What is segmentation?

Image Segmentation is a process in machine learning that’s used to partition an image into distinct regions that 
contain pixels with similar attributes. It can locate the objects in an image and color-code each pixel that 
represents a particular object. It can also color-code each pixel that belongs to a certain class or color-code 
each instance of an object that belongs to the same class.

11. What is y_range used for? When do we need it?

The Y Range is a parameter that's used in Fast.ai to instruct the framework to predict numerical values instead of 
categorical values. It limits the values that are predicted for the dependent variable when performing regression 
which normalizes the data. It also manually specifies the maximum and minimum values which forces the model to 
output values within that range.

12. What are "hyperparameters"?

The Hyperparameter is a variable that's used in machine learning to tune the model to make accurate predictions. It 
sets a value for parameters like learning rate, number of epochs, hidden layers, and activation functions which 
controls the training process. It also must be set manually before the training begins and significantly impacts 
the performance of the model.

13. What's the best way to avoid failures when using AI in an organization?

The best way to avoid failure when introducing artificial intelligence into an organization is to understand and 
use validation and test sets. It can greatly reduce the risk of failure by setting aside some data that's separate 
from the data that's given to the external vendor or service provider. It also lets the organization evaluate the 
true performance of the model with the test data.

14. What is a p value?

A P-value is a measure of the percentage of time we would we see a relationship by chance.

15. What is a prior?

A prior belief/example/hypothesis.

16. Provide an example of where the bear classification model might work poorly in production, due to structural or 
style differences in the training data.

Suppose we would deploy our bear classificator in one of the park in order to warn people about bears nearby. The 
images that bear might appear might different to that seen in training example, for example he might hide under 
the leaves. Might not be fully exposed. Might be in water. If we just had 'clean' images the model could have 
problems with that.

17. Where do text models currently have a major deficiency?

Text models still struggle to produce factually correct responses when asked questions about factual information. 
It can generate responses that appear compelling to the layman but are entirely incorrect. It can also be 
attributed to the challenges in natural language processing that are related to accuracies such as contextual words, 
homonyms, synonyms, sarcasm, and ambiguity.

18. What are possible negative societal implications of text generation models?

The negative societal implications of text generation models are fake news and the spread of disinformation. It 
could be used to produce compelling content on a massive scale with far greater efficiency and lower barriers to 
entry. It could also be used to carry out socially harmful activities that rely on text such as spam, phishing, 
abuse of legal and government processes, fraudulent academic essay writing, and social engineering pretexting.

19. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative 
to automating a process?

The best alternative to artificial intelligence is augmented intelligence which expects humans to interact closely 
with the models. It can make humans 20 times more productive than using strictly manual methods. It can also produce 
more accurate processes than using strictly humans.

20. What kind of tabular data is deep learning particularly good at?

Deep learning is particularly good at analyzing tabular data that contains columns with plain text and 
high-cardinality categorical variables which have many possible values. It can outperform popular machine learning 
algorithms under these conditions. It also takes longer to train, is harder to interpret, involves hyperparameter 
tuning, and requires GPU hardware.

21. What's a key downside of directly using a deep learning model for recommendation systems?

A key downside of recommendation systems is that nearly all deep learning models only recommend products the user 
might like rather than products they might need or find useful. It only recommends similar products based on their 
purchase history, product sales, and product ratings. It also can't recommend novel products that haven't been 
discovered by many users yet.

22. What are the steps of the Drivetrain Approach?

The Drivetrain Approach is a framework that’s used in machine learning to design a system that can solve a complex 
problem. It uses data to produce actionable outcomes rather than just generate more data in the form of predictions. 
It also uses the following 4-step process to build data products:
* Define a clear outcome you are wanting to achieve
* Identify the levers you can pull to influence the outcome
* Consider the data you would need to produce the outcome
* Determine the models you can build to achieve the outcome

23. How do the steps of the Drivetrain Approach map to a recommendation system?

The outcome is to capture additional sales by recommending products to customers that wouldn't have purchased 
without the recommendation. The lever is the method that’s used to choose the recommendations that are shown to 
customers. The data is collected to identify the recommendations that cause new sales which require conducting 
randomized experiments that test a wide range of recommendations for a wide range of customers.

The model is actually two models that predict the purchase probability for products based on whether customers were 
shown the recommendation. It computes the difference between the purchase probabilities to decide the best 
recommendations to display. It also accounts for customers that ignore recommendations and would've purchased 
without the recommendation.

24. Create an image recognition model using data you curate, and deploy it on the web.

The textbook recommends deploying the initial prototype of an application as an interactive Jupyter Notebook 
using Binder. It allows users to create sharable notebooks that can be accessed with a single link. It also 
assigns a virtual machine to run the application which allocates the storage space to store all the files that are 
needed to run the Jupyter Notebook in the cloud.

25. What is DataLoaders?

Data Loader is a class that's used in PyTorch to preprocess the data from the dataset into the format that's 
needed by the model. It specifies the dataset to load and customizes how the dataset gets loaded. It also mostly 
gets used for batching the data, shuffling the data, and loading the data in parallel.

26. What four things do we need to tell fastai to create DataLoaders?

Data Block is a class that's used in Fastai to build datasets and data loaders objects. It must specify the blocks, 
get_items, splitter, and get_y parameters to build the data loaders object. It can also use various combinations of 
the parameters to build different types of data loaders for deep learning models.
* blocks: Sets the functions for the input (left) and output (right) type
* get_items: Sets the input file paths using the get_image_files function
* splitter: Sets the function for splitting the training and validation sets
* get_y: Sets the labels function that extracts the labels from the dataset

27. What does the splitter parameter to DataBlock do?

Splitter is a parameter in the DataBlock class that’s used in Fastai to split the dataset into subsets. It sets the 
function that defines how to split the dataset into training and validation subsets. It also mostly uses the 
RandomSplitter function to randomly split the data but there are nine ways to split the data.

28. How do we ensure a random split always gives the same validation set?

Random Seed is a number that's used in machine learning to initialize the random number generator. It enables the 
random number generator to produce weights with the same sequence of numbers. It also lets users train the model with 
the same code, data, and weights to produce similar results.

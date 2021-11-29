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

Rather than just jumping to a classifier, we can create an IMDB language model first then the classifier.

## Language model from scratch

A word is basically a categorical variable.

1. Make a list of all possible levels of that categorical variable (vocab).
2. Replace each level with its index in the vocab.
3. Create an embedding matrix for this containing a row for each level.
4. Use this embedding matrix as the first layer of a neural network.

Terms: 
* Tokenization
  * Convert text into list of words.
* Numericalization
  * Make a list of unique words and convert them into a number by looking up its index in the vocab.
* Language model data loader
  * Create a data loader. FastAI comes with `LMDataLoader`.
* Language model
  * New model that handles input lists which could be arbitrarily big or small. We will use recurrent neural networks.

## Tokenization

Tokenization has three main approaches: 
* Word-based
  * Split a sentence on spaces, as well as applying language specific rules to try to separate parts of meaning
* Subword based
  * Split words into smaller parts, based on most commonly occurring substrings.
  * Example: "occasion" -> "o", "c", "ca", "sion".
* Character based
  * Split a sentence into its individual characters

* Token
  * One element of a list created by the tokenization process.
  * Could be a word, a subword, or a single character.

## Word tokenizer

FastAI provides an interface to external tokenizers. Default word tokenizer is "spaCy".

```
from fastai.text.all import *
path = untar_data(URLs.IMDB)
files = get_text_files(path, folders = ['train', 'test', 'unsup'])
txt = files[0].open().read(); txt[:75]

spacy = WordTokenizer()
toks = first(spacy([txt]))
first(spacy(['The U.S. dollar $1 is $1.00.']))

tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 31))

defaults.text_proc_rules

coll_repr(tkn('&copy;   Fast.ai www.fast.ai/INDEX'), 31)
```

## Subword tokenizer

```
txts = L(o.open().read() for o in files[:2000])

def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])

subword(1000)
subword(200)
subword(10000)
```

## Question: how can we determine if pre-trained model is suitable for downstream task?

If it's the same language, it's almost always sufficient to use Wikipedia. You don't need corpus specific trained 
models.

## Numericalization

Make a list of unique words and convert them into a number by looking up its index in the vocab.

```
toks = tkn(txt)
print(coll_repr(tkn(txt), 31))

toks200 = txts[:200].map(tkn)
toks200[0]

num = Numericalize()
num.setup(toks200)
coll_repr(num.vocab,20)

nums = num(toks)[:20]; nums

' '.join(num.vocab[o] for o in nums)
```

## Creating batches for language model

```
stream = "In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while."
tokens = tkn(stream)
bs,seq_len = 6,15
d_tokens = np.array([tokens[i*seq_len:(i+1)*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))

bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15:i*15+seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))

bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+seq_len:i*15+2*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))

bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+10:i*15+15] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```

## LMDataLoader

LMDataLoader is a utility to create batched for a language model. Makes it easy.

```
nums200 = toks200.map(num)
dl = LMDataLoader(nums200)
x,y = first(dl)
x.shape,y.shape
' '.join(num.vocab[o] for o in x[0][:20])
' '.join(num.vocab[o] for o in y[0][:20])
```

## Creating language model data with DataBlock

```
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)

dls_lm.show_batch(max_n=2)
```

## Fine-tuning a language model

```
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(1, 2e-2)
```

## Saving and loading models

```
learn.save('1epoch')
learn = learn.load('1epoch')

learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)

learn.save_encoder('finetuned')
```

## Question: Can language models learn meaning?

Yes, machine learning means it learns these rules on its own. Language models in practice are good 
at understanding nuances in language.

## Text generation with language model

```
TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
print("\n".join(preds))
```

```
i liked this movie because of its style and music , and Roman Catholic folklore and interest in books , and the film End of the Century ( 1995 ) . It now has a number of legal issues and
i liked this movie because it was an adaptation of the novel of the same name . Though the film was not specifically designed to be a Holocaust , the author cited the film as a way to establish Christian Christian
```

## Creating classification model

```
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)

dls_clas.show_batch(max_n=3)
```

## Question: Is stemming and lemmatisation still used in practice?

No. It's an outdated approach. Word stems tell us something, so we don't remove them. Used to be used pre-deep learning.

## Handling different sequence lengths

```
nums_samp = toks200[:10].map(num)
nums_samp.map(len)
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 
                                metrics=accuracy).to_fp16()
learn = learn.load_encoder('finetuned')
```

## Fine-tuning classifier

```
learn.fit_one_cycle(1, 2e-2)

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
```

## Questions

It's amazing that a model that can predict the next word of a sentence also classify language (95.1% accuracy).

To do data augmentation on text, you would Google and read papers 
("Unsupervised Data Augmentation for Consistency Training"). Tricks like translating sentences into multiple languages.

## Ethics and risks associated with text generation language models

Misinformation can be increased through deep learning models, ex. GPT2. SciPy keynote lecture.

## Language model from scratch

Will cover how to create a complex architecture: recurring neural network.

## Question: Are there model interpretability tools for language models?

There are tools. We won't cover them in this course. There are PyTorch libraries for this. Technical manuals, grammar
textbooks, and such is not a bad idea to train your model.

## Preparing the dataset for RNN: tokenization and numericalization

Jeremy Howard made the "Human Numbers" data set for RNNs.

```
from fastai.text.all import *
path = untar_data(URLs.HUMAN_NUMBERS)

Path.BASE_PATH = path
path.ls()

lines = L()
with open(path/'train.txt') as f: lines += L(*f.readlines())
with open(path/'valid.txt') as f: lines += L(*f.readlines())
lines

text = ' . '.join([l.strip() for l in lines])
text[:100]

tokens = text.split(' ')
tokens[:10]

vocab = L(*tokens).unique()
vocab

word2idx = {w:i for i,w in enumerate(vocab)}
nums = L(word2idx[i] for i in tokens)
nums
```

## Defining a simple language model

```
L((tokens[i:i+3], tokens[i+3]) for i in range(0,len(tokens)-4,3))
seqs = L((tensor(nums[i:i+3]), nums[i+3]) for i in range(0,len(nums)-4,3))
bs = 64
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(seqs[:cut], seqs[cut:], bs=64, shuffle=False)
```

## Question: Can you speed up fine-tuning the NLP model?

10+ mins per epoch is tough. You don't normally need to fine-tune that often. Work is more at the classifier stage.

## Simple language model continued

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
        
learn = Learner(dls, LMModel1(len(vocab), 64), loss_func=F.cross_entropy, 
                metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)

n,counts = 0,torch.zeros(len(vocab))
for x,y in dls.valid:
    n += y.shape[0]
    for i in range_of(vocab): counts[i] += (y==i).long().sum()
idx = torch.argmax(counts)
idx, vocab[idx.item()], counts[idx].item()/n
```

## Recurrent neural networks (RNN)

```
class LMModel2(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        h = 0
        for i in range(3):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
        return self.h_o(h)

learn = Learner(dls, LMModel2(len(vocab), 64), loss_func=F.cross_entropy, 
                metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)
```

~47% accuracy.

## Improving our RNN

```
class LMModel3(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        self.h = 0
        
    def forward(self, x):
        for i in range(3):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
        out = self.h_o(self.h)
        self.h = self.h.detach()
        return out
    
    def reset(self): self.h = 0
```

## Back propagation through time

Gradients need to be calculated through every layer. To aid in this calculation, we do `self.h.detach()` which
throws away the gradient history. This is called truncated back propagation.

## Ordered sequences and callbacks

```
m = len(seqs)//bs
m,bs,len(seqs)

def group_chunks(ds, bs):
    m = len(ds) // bs
    new_ds = L()
    for i in range(m): new_ds += L(ds[i + m*j] for j in range(bs))
    return new_ds
    
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(
    group_chunks(seqs[:cut], bs), 
    group_chunks(seqs[cut:], bs), 
    bs=bs, drop_last=True, shuffle=False)

learn = Learner(dls, LMModel3(len(vocab), 64), loss_func=F.cross_entropy,
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(10, 3e-3)
```

## Creating more signal for model

```
sl = 16
seqs = L((tensor(nums[i:i+sl]), tensor(nums[i+1:i+sl+1]))
         for i in range(0,len(nums)-sl-1,sl))
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(group_chunks(seqs[:cut], bs),
                             group_chunks(seqs[cut:], bs),
                             bs=bs, drop_last=True, shuffle=False)

[L(vocab[o] for o in s) for s in seqs[0]]

class LMModel4(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        self.h = 0
        
    def forward(self, x):
        outs = []
        for i in range(sl):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
            outs.append(self.h_o(self.h))
        self.h = self.h.detach()
        return torch.stack(outs, dim=1)
    
    def reset(self): self.h = 0

def loss_func(inp, targ):
    return F.cross_entropy(inp.view(-1, len(vocab)), targ.view(-1))

learn = Learner(dls, LMModel4(len(vocab), 64), loss_func=loss_func,
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)
```

## Multilayer RNN

```
class LMModel5(Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, batch_first=True)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = torch.zeros(n_layers, bs, n_hidden)
        
    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(res)
    
    def reset(self): self.h.zero_()

learn = Learner(dls, LMModel5(len(vocab), 64, 2), 
                loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)
```

~54% accuracy

## Exploding and vanishing gradients

Exploding or disappearing activations happen when each time a matrix multiplication is being done. If you do it 
enough times, you can end up with very large or very small results. Floating point numbers become less and less 
accurate the further away the numbers get from zero. To avoid exploding or disappearing gradients we use LSTM.

## LSTM

LSTM (long short-term memory) is an architecture with two hidden states. The first hidden state is already used
by the RNN to retain the output layer info to predict the next token. The second hidden state is used for keeping
long short-term memory while the hidden state focuses on the next token.

```
class LSTMCell(Module):
    def __init__(self, ni, nh):
        self.forget_gate = nn.Linear(ni + nh, nh)
        self.input_gate  = nn.Linear(ni + nh, nh)
        self.cell_gate   = nn.Linear(ni + nh, nh)
        self.output_gate = nn.Linear(ni + nh, nh)

    def forward(self, input, state):
        h,c = state
        h = torch.cat([h, input], dim=1)
        forget = torch.sigmoid(self.forget_gate(h))
        c = c * forget
        inp = torch.sigmoid(self.input_gate(h))
        cell = torch.tanh(self.cell_gate(h))
        c = c + inp * cell
        out = torch.sigmoid(self.output_gate(h))
        h = out * torch.tanh(c)
        return h, (h,c)
```

```
class LSTMCell(Module):
    def __init__(self, ni, nh):
        self.ih = nn.Linear(ni,4*nh)
        self.hh = nn.Linear(nh,4*nh)

    def forward(self, input, state):
        h,c = state
        # One big multiplication for all the gates is better than 4 smaller ones
        gates = (self.ih(input) + self.hh(h)).chunk(4, 1)
        ingate,forgetgate,outgate = map(torch.sigmoid, gates[:3])
        cellgate = gates[3].tanh()

        c = (forgetgate*c) + (ingate*cellgate)
        h = outgate * c.tanh()
        return h, (h,c)
```

```
t = torch.arange(0,10)
t.chunk(2)
```

## Questions

The identity matrix is the matrix where if you multiply by that same number, you get the original number. 
One popular approach is to start with an identity matrix to avoid gradient explosions.

You can quickly check if gradients are exploding/disappearing when you calculate them. FastAI has `ActivationStats`
to view this information.

## Regularisation using Dropout
## AR and TAR regularisation
## Weight tying
## TextLearner
## Conclusion

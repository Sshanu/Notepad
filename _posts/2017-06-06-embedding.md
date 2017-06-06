---
layout: post
title: "Word Embedding"
description: 
headline: 
modified: 2017-03-13
category: nlp
tags: []
comments: true
blog: true
mathjax: 
---
Word embedding is a vector representations of words. 

In this post we are using [GloVe](https://nlp.stanford.edu/projects/glove/) as pretrained embedding for builing vocabulary of words.

For generating a **headline** from a news article, first words have to be tokenized from embedding matrix.
Dataset will consists of headlines and descriptions of news. [Signal media dataset](http://research.signalmedia.co/newsir16/signal-dataset.html) and [BBC dataset](http://mlg.ucd.ie/datasets/bbc.html) are good.

Once dataset is downloaded, save it in a python pickle file as a tuple of **head** and **desc** where head is headline and desc is description of news.

Load the dataset
{% highlight python %}
import _pickle as cPickle
f = open('data.pkl', 'rb')
head, desc = cPickle.load(f)

# Lowercase the words
head = [h.lower() for h in head]
desc = [d.lower() for d in desc]
{% endhighlight %}

For building the vocabulary first we count the words and then sort it.
**vocab** consists of all the different words in descending order of their count.
{% highlight python %}
from collections import Counter
from itertools import chain
lst= head + desc
vocabcount = Counter(w for txt in lst for w in txt.split())
vocab = list(map(lambda x:x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))
{% endhighlight %}

Word distribution in headlines and description
{% highlight python %}
import matplotlib.pyplot as plt
%matplotlib inline
# log scale and 'clip' will map all negative values a very small positive one
plt.plot([vocabcount[w] for w in vocab])
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log", nonposy='clip')
plt.title('word distribution in headlines and discription')
plt.xlabel('rank')
plt.ylabel('total appearances');
{% endhighlight %}
![Word Distribution]({{ site.url }}/images/word_distribution.png)

Vocab size is 40000 and embedding dimension is 100.
End of sentence and words which are out of vocab are indexed 1 and 0 respectively.
{% highlight python %}
vocab_size = 40000
embedding_dim = 100

empty = 0  # index of no data
eos = 1    # index of end of sentence
start_idx = eos+1 # index of first real word
{% endhighlight %}

Indexing words to vectors and vectors to words.
{% highlight python %}
# Word to Vector
word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
word2idx['<empty>'] = empty
word2idx['<eos>'] = eos
# Vector to word
idx2word = dict((idx,word) for word,idx in word2idx.items())
{% endhighlight %}

Download the [Glove Embedding](http://nlp.stanford.edu/data/glove.6B.zip) then read it.
* **glove_n_symbols** is number of different symbols or words.
* **glove_embedding_weights** is the glove embedding matrix.
* **glove_index_dict** indexes words to vectors.
{% highlight python %}
import os
import numpy as np

fname = 'glove.6B.%dd.txt'%embedding_dim
dir_path = os.getcwd()
glove_name = os.path.join(dir_path, fname)
glove_n_symbols = !wc -l {glove_name}
glove_n_symbols = int(glove_n_symbols[0].split()[0])
glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
global_scale =.1
with open(glove_name, 'r') as f:
    i=0
    for l in f:
        l = l .strip().split()
        w = l[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i,:] = list(map(float, l[1:]))
        i += 1
{% endhighlight %}

Lower casing the words and then indexing them to **glove_index_dict**.
{% highlight python %}
for w, i in glove_index_dict.items():
    w = w.lower()
    if w not in glove_index_dict:
        glove_index_dict[w] = i
{% endhighlight %}

Generating a random embedding with same scale as glove
{% highlight python %}
seed = 42
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
{% endhighlight %}

Initializing the embedding matrix using Glove embedding matrix.
{% highlight python %}
c = 0
x = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is None and w.startswith('#'): # glove has no hastags (I think...)
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
        x += 1
    if g is not None:
        embedding[i,:] = glove_embedding_weights[g,:]
        c+=1
{% endhighlight %}

**word2glove** is collection of all the words in the glove vocabulary.
{% highlight python %}
word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    else:
        continue
    word2glove[w] = g
{% endhighlight %}

Lots of word in the full vocabulary (word2idx) are outside vocab_size.
Build an alterantive which will map them to their closest match in glove but only if the match is good enough.
First normalize the embedding and choose **nb_unkown_words** as they are the last words inside the embedding matrix which are considered to be outisde.
{% highlight python %}

glove_thr = 0.5 # used to compare the closeness
# Normalizing the embedding matrix 
normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]
np.shape(normed_embedding)
nb_unknown_words = 500
glove_match = []
for w, idx in word2idx.items():
    if idx >= vocab_size - nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx,:].copy()
        # find row in embedding that has highest score with gwieght
        gweight /= np.sqrt(np.dot(gweight, gweight))
        np.shape(gweight)
        score = np.dot(normed_embedding[: vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr :
                break
            if idx2word[embedding_idx] in word2glove:
                glove_match.append((w, embedding_idx, s))
                break
            score[embedding_idx] = -1
# Sort on basis of score
glove_match.sort(key = lambda x: -x[2])
{% endhighlight %}

**glove_idx2idx** is a lookup table of index of outside words to index of inside words.
{% highlight python %}
glove_idx2idx = dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in glove_match)
{% endhighlight %}

Then vectorize the headlines and descriptions.
{% highlight python %}
Y = [[word2idx[token] for token in headline.split()] for headline in head]
X = [[word2idx[token] for token in d.split()] for d in desc]
{% endhighlight %}

Visualize X and Y 
{% highlight python %}
plt.hist(list(map(len,Y)),bins=50);
plt.hist(list(map(len,X)),bins=50);
{% endhighlight %}
<div class="row">
    <div class="small-12 medium-6 columns">
		<figure>
		  <img src="{{ site.url }}/images/Y.png">
		  <figcaption>Distribution of Y</figcaption>
		</figure>
	</div>
	<div class="small-12 medium-6 columns">
		<figure>
			<img src="{{ site.url }}/images/X.png">
	  		<figcaption>Distribution of X</figcaption>
  		</figure>
	</div>
</div>
  




Save the embedding and data(X,Y) as a pkl files
{% highlight python %}
with open('vocab-embedding.pkl', 'wb') as f:
    cPickle.dump((embedding, idx2word, word2idx, glove_idx2idx), f, -1)
with open('vocab-embedding.data.pkl', 'wb') as f:
    cPickle.dump((X,Y), f, -1)    
{% endhighlight %}


[Github Repo](https://github.com/udibr/headlines)


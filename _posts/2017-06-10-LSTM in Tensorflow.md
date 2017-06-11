---
layout: post
title: "LSTM in Tensorflow"
description: 
headline: 
modified: 2017-03-13
category: Tensorflow
tags: []
comments: true
blog: true
mathjax: 
---
<h2>Long Short Term Memory Networks</h2>

 ``LSTM`` are special kind of RNN, capable of learning long-term dependencies. Tensorflow is not very well documented hence many people get trouble implenting RNN and LSTM.
Please read [this article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for understanding well.
This post follows from [Erik Hallstrom article about RNN](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767).

LSTM Network has structure consisting of gates and states.

![LSTM Network]({{ site.url}}/images/lstm_chain.png)

![LSTM Network]({{ site.url }}/images/lstm.png)

*`i`*  is the Input Gate,  *`f`* is the Forget Gate,  *`o`* is the Output Gate,  *`c`* is Cell State and  *`s`* is Hidden State.

<h2> Setup</h2>
* **`Truncated length`** is the length of the sequence to feed in.
* **`State Size`** is the dimension of the hidden and cell states.
{% highlight python linenos %}
truncated_length = 10
state_size = 4
batch_size = 100
num_classes = 2
{% endhighlight %}

<h2>Variables and Placeholders</h2>
**`data`** and **`labels`** are the placeholders for input sequence and output sequence.
{% highlight python linenos %}
data = tf.placeholder(tf.float32, [batch_size, truncated_length])
labels = tf.placeholder(tf.int32, [batch_size, truncated_length])
{% endhighlight %}

Placeholders for **`Hidden States`** and **`Cell States`**. 
{% highlight python linenos %}
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
{% endhighlight %}

In Tensorflow states are feed to LSTM in form of tuples, delcared as **`LSTMStateTuple`** in which first element is cell state and the second is the hidden state.
{% highlight python linenos %}
init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
{% endhighlight %}

These Variables are for final linear layer in which hidden states are fed as input.
{% highlight python linenos%}
W = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b = tf.Variable(np.zeros((1, num_classes)) ,dtype=tf.float32)
{% endhighlight%}

<h2>Unstacking</h2>
We want to split the batch data into adjacent time-steps.
{% highlight python linenos %}
input_series = tf.split(data,truncated_length, 1)
label_series = tf.unstack(labels, axis=1)
{% endhighlight %}

<h2>LSTM Cell</h2>
Tensorflow provide s wrapper to create various structures like RNN, LSTM, GRU. So here we are using **`BasicLSTMCell`** which only requires state size as an arugment. 
We are using **`static_rnn`** for forward passing the input and states in LSTM Cell as it only requries input series and LSTM State Tuple and return series of states in every time step and current state.
{% highlight python linenos %}
cell = tf.contrib.rnn.BasicLSTMCell(state_size)
state_series, current_state = tf.contrib.rnn.static_rnn(cell, input_series, init_state)
{% endhighlight %}

<h2>LOSS</h2>
{% highlight python linenos %}
logit_series = [tf.matmul(state, W) + b for state in state_series]
prediction_series = [tf.nn.softmax(logit) for logit in logit_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logit, labels=label) for logit, label in zip(logit_series, label_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(.3).minimize(total_loss)
{% endhighlight %}

<h2>Training</h2>
{% highlight python linenos%}
for epoch in range(num_epochs):
        current_cell_state = np.zeros((batch_size, state_size))
        current_hidden_state = np.zeros((batch_size, state_size))

        _total_loss, _train_step, _current_state, _prediction_series = sess.run(
            [total_loss, train_step, current_state, prediction_series],
            feed_dict={
                data: input_x,
                labels: input_y,
                cell_state: current_cell_state,
                hidden_state: cuurent_hidden_state
            })
        current_cell_state, current_hidden_state = _current_state
{% endhighlight %}

**Full Code** is [here](https://gist.github.com/Sshanu/46c8e703608b3f09892b38a6c403dfde)

<h2>Next Blog</h2>
[Multi Layer LSTM in Tensorflow]({{site.url}}/tensorflow/multilayer-lstm)
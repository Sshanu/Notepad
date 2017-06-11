---
layout: post
title: "Multilayer LSTM"
description: 
headline: 
modified: 2017-06-11
category: Tensorflow
tags: []
comments: true
blog: true
mathjax: 
---
<h2>Multi Layer LSTM in Tensorflow</h2> 

![LSTM Network]({{ site.url }}/images/multilayer_lstm.png)

In [previous post]({{ site.url }}/tensorflow/LSTM-in-Tensorflow) we implmented LSTM in Tensorflow. In this post we will make multilayer LSTM architecture. In Multi Layer LSTM, input to a next LSTM-layer will be the hidden state of previous LSTM-layer.

For every layer of LSTM we have to create hidden and cell state placeholders, so modify this :
{% highlight python linenos %}
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
{% endhighlight %}

with a single placeholder containing states for all the layers:
{% highlight python linenos%}
num_layers = 3
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)])
{% endhighlight %}

In forward pass, replace these lines:
{% highlight python linenos %}
cell = tf.contrib.rnn.BasicLSTMCell(state_size)
state_series, current_state = tf.contrib.rnn.static_rnn(cell, input_series, init_state)
{% endhighlight %}

Tensorflow have a wrapper **`MultiRNNCell`** for implementing multilayer lstms.
Here **`static_rnn`** will return the hidden state series of last layer and current state tuple for all the layers.
{% highlight python linenos%}
def lstm_cell():
    cell = tf.contrib.rnn.LSTMCell(state_size)
    return cell

rnn_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple = True)
state_series, current_state = tf.contrib.rnn.static_rnn(
    rnn_cells, input_series, initial_state=rnn_tuple_state)
{% endhighlight%}

In training replace these :
{% highlight python linenos%}

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
with these lines:
{% highlight python linenos%}
_current_state = np.zeros([num_layers, 2, batch_size, state_size])

_total_loss, _train_step, _current_state, _prediction_series = sess.run(
    [total_loss, train_step, current_state, prediction_series],
    feed_dict={
        data: input_x,
        labels: input_y,
        init_state:_current_state
            })
{% endhighlight %}
For Multilayer LSTM full code is [here](https://gist.github.com/Sshanu/e06169498819aaaa03dfc9d36e58f630)
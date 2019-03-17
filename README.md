# KENN: Knowledge Enhanced Neural Networks
KENN (Knowledge Enhanced Neural Networks) is a library for python 2.7 built on top of TensorFlow that permit to enhance neural networks models with logical constraints. It does so by adding a new final layer, called **Knowledge Enhancer (KE)**, to the existing neural network. The KE change the orginal predictions of the standard neural network enforcing the satisfaction of the constraints. Additionally, it contains **clause weights**, learnable parameters associated to the constraints that represent their strength. 

## Installation
KENN can be installed using pip:
```
pip install kenn
```

## Getting started
To use KENN, you can start from existing TensorFlow code. Few changes must be done to add the logical clauses.
Here a simple example, where we higlighted the 5 changes that are typically needed (for simplicity, we omit most of the standard TensorFlow code):


```python

import tensorflow as tf

# *** 1 ***
from kenn import Knowledge_base as kb

# *** 2 ***
clauses = kb.read_knowledge_base(kb_file_name)

# Standard TensorFlow code
# ....

# Calculate preactivations
z = tf.matmul(h, w) + bias

# Standard code:
# y_hat = tf.nn.sigmoid(z)

# *** 3 ***
_, y_hat = kb.knowledge_enhancer(z, clauses)

# Loss definition 
loss = tf.losses.mean_squared_error(y_hat, y)

# optimizer
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# *** 4 ***
clauses_clip_ops = kb.clip_weigths(clauses)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    _ = sess.run(train_step)
    
    # *** 5 ***
    sess.run(clauses_clip_ops)

# Evaluations and standard operations ...
```


## Knowledge Base file format

## API documentation

## License
Copyright (c) 2019, Daniele Alessandro, Serafini Luciano
All rights reserved.

Licensed under the BSD 3-Clause License.

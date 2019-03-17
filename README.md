# KENN: Knowledge Enhanced Neural Networks
KENN (Knowledge Enhanced Neural Networks) is a library for python 2.7 built on top of TensorFlow that permit to enhance neural networks models with logical constraints (clauses). It does so by adding a new final layer, called **Knowledge Enhancer (KE)**, to the existing neural network. The KE change the orginal predictions of the standard neural network enforcing the satisfaction of the constraints. Additionally, it contains **clause weights**, learnable parameters associated to the constraints that represent their strength. 

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

### Example explained
In the previous example, we applies 5 changes to the standard TensorFlow code. Following, the details.

#### 1. Import knowledge_base module
The first change is trivial, we need to import the library:
```python
from kenn import Knowledge_base as kb
```

#### 2. Read the knowledge base file
```python
clauses = kb.read_knowledge_base(kb_file_name)
```

The `read_knowledge_base` function takes as input the path of the file containing the logical constraints. Following, an example of knowledge base file:

```
Dog,Cat,Animal,Car,Truck,Chair

nDog,Animal
nCat,Animal
nDog,nCat
nCar,Animal
nAnimal,Dog,Cat
```

The first row contains a list of predicates separated with a comma with no spaces. Each predicate must start with a capital letter.
The second row must be empty.
Other rows contain the clauses.

Each clause is in a separate row and must be written respecting this properties:
1. logical disjunctions are represented with commas
1. if a literal is negated, it must be precedeed by the lowercase 'n'
1. they must contain only predicates specified in the first row
1. there shouldn't be spaces

For example, the third line represents the clause <a href="https://www.codecogs.com/eqnedit.php?latex=\lnot&space;Dog&space;\lor&space;Animal" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lnot&space;Dog&space;\lor&space;Animal" title="\lnot Dog \lor Animal" /></a> and tells us that a dog should also be an animal. A more interesting clause is the last one, that tells us that in our domain only cats and dogs are animals.

#### 3. Knowledge Enhancer (KE)
This is the most relevant change. Usually, in the last layer of a neural network, we apply an activation function to the preactivations calculated by previous layer.
```python
y_hat = tf.nn.sigmoid(z)
```
Here, instead, we feed the preactivations to the knowledge enhancer:
```python
_, y_hat = kb.knowledge_enhancer(z, clauses)
```

The `knowledge_enhancer` function takes as input the preactivations of the final layer and the set of clauses previously generated by the `read_knowledge_base` function (see 2.). It defines a new layer that changes the results by enforcing the clauses satisfaction. More precisely, it returns a tuple containing the preactivations (usefull when using the `cross_entropy_with_logits` loss function) and final activations of the KE layer.

At this point, we can use the output of the `knowledge_enhancer` as we would have used the original activations: we can define the loss function directly on such predictions.

**NB:** in order to work properly, the preactivations tensor given as input to `knowledge_enhancer` should be a matrix whose rows represent a possible grounding. The columns represent the predicates of the knowledge base and should be in the same order as specified in the first row of knowledge base file.

#### 4-5. Clauses weights clipping
Another change that is needed is to define the clauses_clip_ops.
```python
clauses_clip_ops = kb.clip_weigths(clauses)
```

Such operation set to zero the clause weights that became negative. It should be called after each train step in order to keep the clauses weights positive:
```python
_ = sess.run(train_step)
sess.run(clauses_clip_ops)
```


## License
Copyright (c) 2019, Daniele Alessandro, Serafini Luciano
All rights reserved.

Licensed under the BSD 3-Clause License.

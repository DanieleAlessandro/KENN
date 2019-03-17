"""clause module"""

import tensorflow as tf
import numpy as np
from delta_functions import softmax


class Clause:
    """Represent a clause in CNF.

    """

    def __init__(self, available_literals, clause_string, initial_weight_value):
        """Initialize the clause.

        :param available_literals: the list of all possible literals in a clause
        :param clause_string: a string representing a conjunction of literals. The format should be:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).

        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.

        An example:
           _:nDog,Animal

        :param initial_weight_value: the initial value to the clause weight. Used if the clause weight is learned
        """

        string = clause_string.split(':')

        self.original_string = string[1]
        self.string = string[1].replace(',', 'v').replace('(', '').replace(')', '')

        with tf.name_scope(self.string):
            if string[0] == '_':
                self.hard_clause = False
                self.clause_weight = tf.Variable(initial_weight_value, name='w')
            else:
                self.hard_clause = True
                self.clause_weight = tf.constant(float(string[0]), name='w')

            literals_list = string[1].split(',')
            self.number_of_literals = len(literals_list)

            self.delta = 0.0

            self.gather_literal_indices = []
            self.scatter_literal_indices = []
            signs = []

            for literal in literals_list:
                value = 1
                if literal[0] == 'n':
                    value = -1
                    literal = literal[1:]

                literal_index = available_literals.index(literal)
                self.gather_literal_indices.append(literal_index)
                self.scatter_literal_indices.append([literal_index])
                signs.append(value)

            self.signs = tf.constant(np.array(signs, dtype=np.float32))

    def clip_weight(self):
        """Clip operation: if the weight is negative it become zero.

        :return: the tensorflow clip operation
        """
        if not self.hard_clause:
            return tf.assign(self.clause_weight, tf.clip_by_value(self.clause_weight, 0.0, 500.0))
        else:
            raise Exception('Clip weight can not be applied on hard constraints')

    def grounded_clause(self, tensor):
        """Find the grounding of the clause

        :param tensor: the tensor containing predicates' pre activations for many entities
        :return: the grounded clause (a tensor with literals truth values)
        """

        with tf.name_scope('groundings'):
            selected_predicates = tf.gather(tensor, self.gather_literal_indices, axis=1)
            clause_matrix = selected_predicates * self.signs

        return clause_matrix

    def clause_enhancer(self, tensor, enhancer_function=softmax):
        """Improve the satisfaction level of the clause.

        :param tensor: the tensor containing predicates' pre-activation values for many entities
        :param enhancer_function: the function that return delta values starting from pre-activations
        :return: delta vector to be summed to the original pre-activation tensor to obtain the higher satisfaction of \
        the clause"""
        with tf.name_scope(self.string + '_enhancer'):
            clause_matrix = self.grounded_clause(tensor)

            delta = self.signs * enhancer_function(clause_matrix) * self.clause_weight

            scattered_delta = tf.scatter_nd(self.scatter_literal_indices, tf.transpose(delta), tf.reverse(tf.shape(tensor), [0]))

        return scattered_delta

    def to_string(self, sess):
        """Return the string representation of the clause

        :param sess: the tensorflow session
        :return: a string representing the clause together with the [learned] clause weight
        """
        w = sess.run(self.clause_weight)
        return str(w) + ':' + self.original_string

    def __str__(self):
        return self.original_string

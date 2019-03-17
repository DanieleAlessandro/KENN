"""knowledge base module.

It manages a knowledge base that consists on a set of clauses.
"""

from Clause import Clause
import tensorflow as tf


def read_knowledge_base(file_name, initial_clause_weight=0.5):
    """Initialize the knowledge base.

    :param file_name: the path of the knowledge base file. The file must contain:
        - a row with the list of predicates
        - a set of constraints. Each constraint is on the form:
        clause_weight:clause

        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).

        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.

        An example:
           _:nDog,Animal

    :param initial_clause_weight: the initial value to the clause weights. Used for the clause precedeed by an
        underscore
    :return: the lists of clauses in the knowledge base
    """
    with open(file_name, 'r') as kb_file:
        unary_literals_string = kb_file.readline()
        kb_file.readline()
        rows = kb_file.readlines()

    u_groundings = unary_literals_string[:-1].split(',')

    unary_clauses = []

    for r in rows:
        unary_clauses.append(Clause(u_groundings, r[:-1], initial_clause_weight))

    return unary_clauses


def clip_weights(clauses):
    """Generates the clip operations to be applied for keeping the clause weights positive

    :param clauses: list of clauses
    :return: the list of clip operations on the set of clauses
    """
    return [c.clip_weight() for c in clauses if not c.hard_clause]


def knowledge_enhancer(tensor, clauses):
    """Improve the satisfaction level of a set of clauses.

    :param tensor: the tensor containing predicates' pre-activation values for many entities
    :return: tuple (preactivation, final predictions)"""
    deltas = []
    for clause in clauses:
        deltas.append(clause.clause_enhancer(tensor))

    final_deltas = tf.transpose(tf.add_n(deltas))

    return tensor + final_deltas, tf.nn.sigmoid(tensor + final_deltas)


def kb_to_string(sess, clauses):
    """Return the string representation of the knowledge base

    :param sess: the tensorflow session
    :param clauses: list of clauses

    :return: a string representing the knowledge base together with the [learned] clause weights
    """
    s = []
    for clause in clauses:
        s.append(clause.to_string(sess) + '\n')

    return s

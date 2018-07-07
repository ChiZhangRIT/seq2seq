"""Module for constructing RNN Cells.

## Base interface for all RNN Cells

https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/rnn/python/ops/core_rnn_cell.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs


RNNCell = rnn_cell_impl.RNNCell


class EmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.
  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self,
               cell,
               embedding_classes,
               embedding_size,
               initializer=None,
               reuse=None):
    """Create a cell with an added input embedding.
    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    super(EmbeddingWrapper, self).__init__(_reuse=reuse)
    rnn_cell_impl.assert_like_rnncell("cell", cell)
    if embedding_classes <= 0 or embedding_size <= 0:
      raise ValueError("Both embedding_classes and embedding_size must be > 0: "
                       "%d, %d." % (embedding_classes, embedding_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run the cell on embedded inputs."""
    with ops.device("/cpu:0"):
      if self._initializer is not None:
        initializer = self._initializer
      elif vs.get_variable_scope().initializer is not None:
        initializer = vs.get_variable_scope().initializer
      else:
        # Default initializer for embeddings should have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

      if isinstance(state, tuple):
        data_type = state[0].dtype
      else:
        data_type = state.dtype

      embedding = vs.get_variable(
          "embedding", # [self._embedding_classes, self._embedding_size],
          initializer=initializer,
          trainable=True,  # change trainable if needed
          dtype=data_type)
      embedded = embedding_ops.embedding_lookup(embedding,
                                                array_ops.reshape(inputs, [-1]))

      return self._cell(embedded, state)

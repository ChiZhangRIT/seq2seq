"""Library for creating sequence-to-sequence models in TensorFlow.

* Full sequence-to-sequence models.
    - embedding_attention_seq2seq: Advanced model with input embedding and the neural attention mechanism; recommended for complex tasks.

https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py#L791
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import copy
import pdb

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from embedding import rnn_cell


def _extract_argmax_and_embed(embedding, output_projection=None, update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.
    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.
    Returns:
        A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None, scope=None):
    """RNN decoder for the sequence-to-sequence model.
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th output
            in order to generate the i+1-st input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
            state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                states can be the same. They are different for LSTM cells though.)
    """
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states, cell, num_symbols, embedding_size, num_heads=1, output_size=None, output_projection=None, feed_previous=False, update_embedding_for_previous=True, dtype=None, scope=None, initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.
    Args:
        decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: tf.nn.rnn_cell.RNNCell defining the cell function.
        num_symbols: Integer, how many symbols come into the embedding.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_size: Size of the output vectors; if None, use output_size.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_symbols] and B has shape
            [num_symbols]; if provided and feed_previous=True, each fed previous
            output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
            used (the "GO" symbol), and all other decoder inputs will be generated by:
                next = embedding_lookup(embedding, argmax(previous_output)),
            In effect, this implements a greedy decoder. It can also be used
            during training to emulate http://arxiv.org/abs/1506.03099.
            If False, decoder_inputs are used as given (the standard decoder case).
        update_embedding_for_previous: Boolean; if False and feed_previous=True,
            only the embedding for the first symbol of decoder_inputs (the "GO"
            symbol) will be updated by back propagation. Embeddings for the symbols
            generated from the decoder itself remain unchanged. This parameter has
            no effect if feed_previous=False.
        dtype: The dtype to use for the RNN initial states (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.
    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
        ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = [v for v in tf.global_variables()
                if v.name == "embedding_attention_seq2seq/rnn/embedding_wrapper/embedding:0"][0]

    with variable_scope.variable_scope("embedding_attention_decoder", dtype=dtype) as scope:

        # embedding = variable_scope.get_variable("embedding",
        #                                         [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
        ]

        return rnn_decoder(
            emb_inp,                    # decoder_inputs
            initial_state,              # initial_state
            cell,                       # cell
            loop_function=loop_function)

        # return attention_decoder(
        #     emb_inp,
        #     initial_state,
        #     attention_states,
        #     cell,
        #     output_size=output_size,
        #     num_heads=num_heads,
        #     loop_function=loop_function,
        #     initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols, embedding_size, num_heads=1, output_projection=None, feed_previous=False, dtype=None, scope=None, initial_state_attention=False, pretrained_embedding_path=None):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape [num_encoder_symbols x input_size]). Then it runs an RNN to encode embedded encoder_inputs into a state vector. It keeps the outputs of this RNN at every step to use for attention later. Next, it embeds decoder_inputs by another newly created embedding (of shape [num_decoder_symbols x input_size]). Then it runs attention decoder, initialized with the last encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Warning: when output_projection is None, the size of the attention vectors and variables will be made proportional to num_decoder_symbols, can be large.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and 
            biases; W has shape [output_size x num_decoder_symbols] and B has shape 
            [num_decoder_symbols]; if provided and feed_previous=True, each fed 
            previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
            of decoder_inputs will be used (the "GO" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_seq2seq". (deprecated)
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states.
        pretrained_embedding_path: path to GloVe embedding.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x num_decoder_symbols] containing the generated
                outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """

    with variable_scope.variable_scope("embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype

        # Encoder
        pdb.set_trace()
        encoder_cell = copy.deepcopy(cell)
        pdb.set_trace()
        init = tf.constant(np.load(pretrained_embedding_path))  # use glove
        encoder_cell = rnn_cell.EmbeddingWrapper(encoder_cell, 
                                                embedding_classes=num_encoder_symbols,
                                                embedding_size=300, 
                                                initializer=init)

        encoder_outputs, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [
            array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
        ]
        attention_states = array_ops.concat(top_states, 1)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                cell,
                num_decoder_symbols,
                embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)

        # # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        # def decoder(feed_previous_bool):
        #     reuse = None if feed_previous_bool else True
        #     with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
        #         outputs, state = embedding_attention_decoder(
        #                             decoder_inputs,
        #                             encoder_state,
        #                             attention_states,
        #                             cell,
        #                             num_decoder_symbols,
        #                             embedding_size,
        #                             num_heads=num_heads,
        #                             output_size=output_size,
        #                             output_projection=output_projection,
        #                             feed_previous=feed_previous_bool,
        #                             update_embedding_for_previous=False,
        #                             initial_state_attention=initial_state_attention)
        #         state_list = [state]
        #         if nest.is_sequence(state):
        #             state_list = nest.flatten(state)
        #         return outputs + state_list

        # outputs_and_state = control_flow_ops.cond(feed_previous,
        #                                           lambda: decoder(True),
        #                                           lambda: decoder(False))
        # outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        # state_list = outputs_and_state[outputs_len:]
        # state = state_list[0]
        # if nest.is_sequence(encoder_state):
        #     state = nest.pack_sequence_as(
        #         structure=encoder_state, flat_sequence=state_list)
        # return outputs_and_state[:outputs_len], state




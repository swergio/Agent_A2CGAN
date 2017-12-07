from tensorflow.contrib.seq2seq.python.ops.helper import GreedyEmbeddingHelper
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

import tensorflow as tf


'''
Embedding Helper for soft implementation.
Using the embedding of the prior Cell output as next stes input instead of the sample IDs.abs

Inherited from GreedyEmbeddingHelper in tensorflow.contrib.seq2seq library
'''

class SoftGreedyEmbeddingHelper(GreedyEmbeddingHelper):
    
    '''
    Overwrite __init__ 
    Adding the startInput (as Logits) as Input
    overwriting the _start_input as softembedding of logits
    '''
    def __init__(self, embedding, start_tokens, end_token, startInput):
        super(SoftGreedyEmbeddingHelper, self).__init__(embedding, start_tokens, end_token)
        self._start_inputs = self._softembedding_fn(startInput)

    ''' 
    Overwrite next_input function 
    use output instead of sampple_ids for next input
    '''
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._softembedding_fn(outputs))
        return (finished, next_inputs, state)

    '''
    Function to calculate the soft embedding based on logits
    '''
    def _softembedding_fn(self,outputs):

        idx = tf.tile(tf.reshape(tf.range(outputs.shape[1].value),[1,outputs.shape[1].value]),[outputs.shape[0].value,1])
        emb = self._embedding_fn(idx)
        out_soft = tf.tile(tf.expand_dims(tf.nn.softmax(outputs),-1),[1,1,emb.shape[2].value])
        next_inputs = tf.reduce_sum(emb * out_soft,axis =1)
        return next_inputs
    





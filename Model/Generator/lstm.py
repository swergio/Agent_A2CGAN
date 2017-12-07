import numpy as np
import tensorflow as tf

from Model.Generator.seq2seqHelper import SoftGreedyEmbeddingHelper


def lstm(xi, s,sentence_size, scope,embeddingScope, nh,env,PROD = False, init_scale=1.0, steps = None, initLenght = None):

    with tf.variable_scope(embeddingScope,reuse=True):
        A = tf.get_variable("A_msg")
    x =  tf.nn.embedding_lookup(A, xi)
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    nbatch,nin,_ = [v.value for v in x.get_shape()]

    if steps is None:
        steps = tf.constant(sentence_size, shape=[nbatch])
    if initLenght is None:
        initLenght = tf.constant(1,shape=[nbatch])
    length_of_sequence =  initLenght

    maxit = tf.reduce_sum(tf.slice(steps,[0],[1]) - tf.slice(initLenght,[0],[1]))
    print(maxit.shape)


    with tf.variable_scope(scope):
        #learnModel
        Lhelper = tf.contrib.seq2seq.TrainingHelper(x, length_of_sequence, time_major=False)
        Loutputs, Lstate, LOlength = DecoderModel(nh,c,h,Lhelper,sentence_size,False)
        print("nh: " + str(nh))
        Lc = Lstate[0]
        Lh = Lstate[1]
        Llogits = Loutputs.rnn_output
        Lsample = Loutputs.sample_id
        idx= tf.stack(axis=1, values=[tf.range(nbatch),LOlength-1])
        lsam = tf.gather_nd(Lsample,idx)

        #predModel
        print("Llogits: " + str(Llogits[:,-1:,:]))
        Phelper = SoftGreedyEmbeddingHelper(A,lsam, 2,startInput = tf.reshape(Llogits[:,-1:,:],[Llogits.shape[0].value,Llogits.shape[2].value]))

        Poutputs, Pstate, POlength = DecoderModel(nh,Lc,Lh,Phelper,maxit,True)
        Plogits = Poutputs.rnn_output
        Psample = Poutputs.sample_id
        Plogits_pad = tf.constant(0,shape=[nbatch,sentence_size,Plogits.shape[2]],dtype=tf.float32)
        Psample_pad = tf.constant(0,shape=[nbatch,sentence_size])
        logits_con = tf.concat(axis=1, values=[Llogits,Plogits, Plogits_pad])
        sample_con = tf.concat(axis=1, values=[Lsample,Psample, Psample_pad])
        logits = tf.slice(logits_con,[0,0,0],[nbatch,sentence_size,logits_con.shape[2].value])
        sample = tf.slice(sample_con,[0,0],[nbatch,sentence_size])  

        mask = tf.sequence_mask(steps,sentence_size )
       
        txt = tf.to_int64(sample) *tf.to_int64(mask)
    return logits, Pstate, mask, txt


def DecoderModel(nh,c,h,helper,max_iteration, reuse):
    scopeName = "lstm"
    with tf.variable_scope(scopeName, reuse = reuse) as sc:
        
        '''
        LSTM CELL
        '''
        #decoder_cell = tf.nn.rnn_cell.LSTMCell(nh) 
        #initial_state = tf.nn.rnn_cell.LSTMStateTuple(c,h)
        '''
        GRU CELL
        '''
        decoder_cell = tf.nn.rnn_cell.GRUCell(nh) 
        initial_state = h
        # Decoder
        projection_layer = tf.layers.BatchNormalization()
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer=projection_layer)
        # Dynamic decoding
        outputs, state, length = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,maximum_iterations = max_iteration, scope = sc)
        state = tf.nn.rnn_cell.LSTMStateTuple(c,state)
    
    return outputs, state, length 
    


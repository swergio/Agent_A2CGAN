'''
This code is based on Taehoon Kim's implementation of the MemN2N model (https://github.com/carpedm20/MemN2N-tensorflow)
The code is under the following license:

The MIT License (MIT)

Copyright (c) 2016 Taehoon Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import tensorflow as tf
from Model.Utility.layers import batch_normalization

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

def memn2n(X, S,vocab_size,embedding_size,hops, env, name = "memn2n",XasOneHot = False, is_training = True):
        message_space = env.message_space
        sentence_size = int(message_space[3])
        encoding = tf.constant(position_encoding(sentence_size,embedding_size[3]), name="encoding")
        init = tf.random_normal_initializer(stddev=0.1)
        if XasOneHot is False:
            X_cty, X_cnr, X_mty, X_msg, X_act = tf.split(X,num_or_size_splits = message_space, axis = 1)
        else:
            #chat type
            X_cty = tf.tile(tf.expand_dims(tf.slice(X_cty,[0,0,0],[-1,-1,vocab_size[0]]),axis= -1),[1,1,1,embedding_size[0]])
            Xi_cty = tf.tile(tf.reshape(tf.range(vocab_size[0]),[1,1,vocab_size[0]]),[X_cty.shape[0],X_cty.shape[1],1])
            #chat number
            X_cnr = tf.tile(tf.expand_dims(tf.slice(X_cnr,[0,0,0],[-1,-1,vocab_size[1]]),axis= -1),[1,1,1,embedding_size[1]])
            Xi_cnr = tf.tile(tf.reshape(tf.range(vocab_size[1]),[1,1,vocab_size[1]]),[X_cnr.shape[0],X_cnr.shape[1],1])
            #message type
            X_mty = tf.tile(tf.expand_dims(tf.slice(X_mty,[0,0,0],[-1,-1,vocab_size[2]]),axis= -1),[1,1,1,embedding_size[2]])
            Xi_mty = tf.tile(tf.reshape(tf.range(vocab_size[2]),[1,1,vocab_size[2]]),[X_mty.shape[0],X_mty.shape[1],1])
            #message text
            X_msg = tf.tile(tf.expand_dims(tf.slice(X_msg,[0,0,0],[-1,-1,vocab_size[3]]),axis= -1),[1,1,1,embedding_size[3]])
            Xi_msg = tf.tile(tf.reshape(tf.range(vocab_size[3]),[1,1,vocab_size[3]]),[X_msg.shape[0],X_msg.shape[1],1])
            #act flag
            X_act = tf.tile(tf.expand_dims(tf.slice(X_act,[0,0,0],[-1,-1,vocab_size[4]]),axis= -1),[1,1,1,embedding_size[4]])
            Xi_act = tf.tile(tf.reshape(tf.range(vocab_size[4]),[1,1,vocab_size[4]]),[X_act.shape[0],X_act.shape[1],1])
        S_cty, S_cnr, S_mty, S_msg, S_act = tf.split(S,num_or_size_splits = message_space, axis = 2)
        
        with tf.variable_scope(name):

            #chat type
            A_cty_0 = tf.concat(axis=0, values=[ init([vocab_size[0], embedding_size[0]]) ])
            C_cty_0 = tf.concat(axis=0, values=[ init([vocab_size[0], embedding_size[0]]) ])
            A_cty_1 = tf.get_variable(initializer = A_cty_0, name="A_cty")
            C_cty = []
            for hopn in range(hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    C_cty.append(tf.get_variable(initializer = C_cty_0, name="C_cty")) 
            #chat number
            A_cnr_0 = tf.concat(axis=0, values=[ init([vocab_size[1], embedding_size[1]]) ])
            C_cnr_0 = tf.concat(axis=0, values=[ init([vocab_size[1], embedding_size[1]]) ])
            A_cnr_1 = tf.get_variable(initializer = A_cnr_0, name="A_cnr")
            C_cnr = []
            for hopn in range(hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    C_cnr.append(tf.get_variable(initializer = C_cnr_0, name="C_cnr")) 
            #message type
            A_mty_0 = tf.concat(axis=0, values=[ init([vocab_size[2], embedding_size[2]]) ])
            C_mty_0 = tf.concat(axis=0, values=[ init([vocab_size[2], embedding_size[2]]) ])
            A_mty_1 = tf.get_variable(initializer = A_mty_0, name="A_mty")
            C_mty = []
            for hopn in range(hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    C_mty.append(tf.get_variable(initializer = C_mty_0, name="C_mty")) 
            #message text
            nil_word_slot = tf.zeros([1, embedding_size[3]])
            A_msg_0 = tf.concat(axis=0, values=[ nil_word_slot, init([vocab_size[3]-1, embedding_size[3]]) ])
            C_msg_0 = tf.concat(axis=0, values=[ nil_word_slot, init([vocab_size[3]-1, embedding_size[3]]) ])
            A_msg_1  = tf.get_variable("A_msg", initializer=A_msg_0)
            C_msg = []
            for hopn in range(hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    C_msg.append(tf.get_variable(initializer = C_msg_0, name="C_msg"))
            #act flag
            A_act_0 = tf.concat(axis=0, values=[ init([vocab_size[4], embedding_size[4]]) ])
            C_act_0 = tf.concat(axis=0, values=[ init([vocab_size[4], embedding_size[4]]) ])
            A_act_1 = tf.get_variable(initializer = A_act_0, name="A_act")
            C_act = []
            for hopn in range(hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    C_act.append(tf.get_variable(initializer = C_act_0, name="C_act"))

        with tf.variable_scope(name) as embeddingScope:
            # Use A_***_1 for thee question embedding as per Adjacent Weight Sharing
            if XasOneHot is False:
                q_emb_msg = tf.nn.embedding_lookup(A_msg_1, X_msg)
                u_0_msg = tf.reduce_sum(q_emb_msg * encoding, 1)
                q_emb_cty = tf.reduce_sum(tf.nn.embedding_lookup(A_cty_1, X_cty),1)
                q_emb_cnr = tf.reduce_sum(tf.nn.embedding_lookup(A_cnr_1, X_cnr),1)
                q_emb_mty = tf.reduce_sum(tf.nn.embedding_lookup(A_mty_1, X_mty),1)
                q_emb_act = tf.reduce_sum(tf.nn.embedding_lookup(A_act_1, X_act),1)
            else:
                q_emb_msg = tf.reduce_sum(tf.nn.embedding_lookup(A_msg_1, Xi_msg) * X_msg,axis= 2)
                u_0_msg = tf.reduce_sum(q_emb_msg * encoding, 1)
                q_emb_cty = tf.reduce_sum(tf.reduce_sum(tf.nn.embedding_lookup(A_cty_1, Xi_cty) * X_cty,axis= 2),1)
                q_emb_cnr = tf.reduce_sum(tf.reduce_sum(tf.nn.embedding_lookup(A_cnr_1, Xi_cnr) * X_cnr,axis= 2),1)
                q_emb_mty = tf.reduce_sum(tf.reduce_sum(tf.nn.embedding_lookup(A_mty_1, Xi_mty) * X_mty,axis= 2),1)
                q_emb_act = tf.reduce_sum(tf.reduce_sum(tf.nn.embedding_lookup(A_act_1, Xi_act) * X_act,axis= 2),1)
            u_0 = tf.concat([q_emb_cty,q_emb_cnr,q_emb_mty,u_0_msg,q_emb_act],axis = 1)
            #batch normalization
            u_0 = batch_normalization(u_0,'bn_u_0',is_training=is_training)
            u = [u_0]

            for hopn in range(hops):
                if hopn == 0:
                    m_emb_A_msg = tf.nn.embedding_lookup(A_msg_1, S_msg)
                    m_A_msg = tf.reduce_sum(m_emb_A_msg * encoding, 2)
                    m_emb_A_cty = tf.reduce_sum(tf.nn.embedding_lookup(A_cty_1, S_cty), 2)
                    m_emb_A_cnr = tf.reduce_sum(tf.nn.embedding_lookup(A_cnr_1, S_cnr), 2)
                    m_emb_A_mty = tf.reduce_sum(tf.nn.embedding_lookup(A_mty_1, S_mty), 2)
                    m_emb_A_act = tf.reduce_sum(tf.nn.embedding_lookup(A_act_1, S_act), 2)
                    m_A = tf.concat([m_emb_A_cty,m_emb_A_cnr,m_emb_A_mty,m_A_msg,m_emb_A_act],axis = 2)
                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A_msg = tf.nn.embedding_lookup(C_msg[hopn - 1], S_msg)
                        m_A_msg = tf.reduce_sum(m_emb_A_msg * encoding, 2)
                        m_emb_A_cty = tf.reduce_sum(tf.nn.embedding_lookup(C_cty[hopn - 1], S_cty), 2)
                        m_emb_A_cnr = tf.reduce_sum(tf.nn.embedding_lookup(C_cnr[hopn - 1], S_cnr), 2)
                        m_emb_A_mty = tf.reduce_sum(tf.nn.embedding_lookup(C_mty[hopn - 1], S_mty), 2)
                        m_emb_A_act = tf.reduce_sum(tf.nn.embedding_lookup(C_act[hopn - 1], S_act), 2)
                        m_A = tf.concat([m_emb_A_cty,m_emb_A_cnr,m_emb_A_mty,m_A_msg,m_emb_A_act],axis = 2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                with tf.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C_msg = tf.nn.embedding_lookup(C_msg[hopn], S_msg)
                    m_C_msg = tf.reduce_sum(m_emb_C_msg * encoding, 2)
                    m_emb_C_cty = tf.reduce_sum(tf.nn.embedding_lookup(C_cty[hopn], S_cty), 2)
                    m_emb_C_cnr = tf.reduce_sum(tf.nn.embedding_lookup(C_cnr[hopn], S_cnr), 2)
                    m_emb_C_mty = tf.reduce_sum(tf.nn.embedding_lookup(C_mty[hopn], S_mty), 2)
                    m_emb_C_act = tf.reduce_sum(tf.nn.embedding_lookup(C_act[hopn], S_act), 2)
                    m_C = tf.concat([m_emb_C_cty,m_emb_C_cnr,m_emb_C_mty,m_C_msg,m_emb_C_act],axis = 2)

                c_temp = tf.transpose(m_C, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)
                u_k = u[-1] + o_k
                #batch normalization 
                u_k = batch_normalization(u_k,'bn_hop_{}'.format(hopn),is_training=is_training)
                u.append(u_k)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(hops)):
                u_k_cty, u_k_cnr, u_k_mty, u_k_msg, u_k_act = tf.split(u_k,num_or_size_splits = embedding_size, axis = 1)
                u_cty  = tf.matmul(u_k_cty, tf.transpose(C_cty[-1], [1,0]))
                u_cnr  = tf.matmul(u_k_cnr, tf.transpose(C_cnr[-1], [1,0]))
                u_mty  = tf.matmul(u_k_mty, tf.transpose(C_mty[-1], [1,0]))
                u_msg  = tf.matmul(u_k_msg, tf.transpose(C_msg[-1], [1,0]))
                u_act  = tf.matmul(u_k_act, tf.transpose(C_act[-1], [1,0]))

                return tf.concat([u_cty,u_cnr,u_mty,u_msg,u_act],axis = 1) , embeddingScope


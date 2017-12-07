import numpy as np
import tensorflow as tf

from Model.Utility.layers import fc, batch_normalization
from Model.Discriminator.encoding import message_encoding



def Discriminator(A,E, env, reuse = False, is_training = True):
    with tf.variable_scope("discriminator", reuse=reuse):
        a_enc =  message_encoding(A,env)
        h = tf.concat([a_enc,E],axis = 1)
        h1 = fc(h, 'h1', int(h.shape[1].value/2))
        #h1_bn = batch_normalization(h1,'h1_bn', is_training=is_training)
        h2 = fc(h1, 'h2',  int(h.shape[1].value/4))
        #h2_bn = batch_normalization(h2,'h2_bn', is_training=is_training)
        d = fc(h2, 'd', 1, act=lambda x:x)
    return d

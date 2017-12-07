import numpy as np
import tensorflow as tf

from Model.Utility.layers import fc, batch_normalization

from Model.Encoder.memn2n import memn2n

def Encoder(X,S,finalSize, env, reuse = False,XasOneHot = False, JustEmbedding = False, is_training=True):
  
    hops = 3
   
    with tf.variable_scope("encoder", reuse=reuse):
        fh, embeddingScope = memn2n(X, S,env.vocab_space,env.embedding_space,hops, env, name = "memn2n",XasOneHot = XasOneHot, is_training = is_training)
        fh = batch_normalization(fh,'BN', is_training = is_training)

        e_mean = fc(fh,'e_mean',finalSize, act=lambda x:x)
        e_log_sigma_sqrt = fc(fh,'e_log_sigma_sqrt',finalSize, act=lambda x:x)
        
        eps = tf.random_normal(shape=(e_mean.shape[0].value, finalSize), mean=0., stddev=1.)
        e =  e_mean + tf.exp(e_log_sigma_sqrt / 2) * eps

        e_stats = (e_mean,e_log_sigma_sqrt)

    return e , e_stats , embeddingScope
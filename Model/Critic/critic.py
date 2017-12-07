import tensorflow as tf
from Model.Utility.layers import fc

def Critic(X, reuse = False):
    with tf.variable_scope("valuefun", reuse=reuse):
            vf = fc(X, 'v', 1, act=lambda x:x)
            v0 = vf[:, 0]

    return v0, vf
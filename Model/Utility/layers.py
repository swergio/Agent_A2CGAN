'''
This code is based on OpenAI's baseline library (https://github.com/openai/baselines) and the included implementation of the A2C model.
OpenAI's code and library is under the following license:

The MIT License

Copyright (c) 2017 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

import numpy as np
import tensorflow as tf
from Model.Utility.ops import ortho_init

def batch_normalization(x, scope, is_training=True):
    with tf.variable_scope(scope):
        h = tf.layers.batch_normalization(x, training = is_training)
        return h

def fc(x,scope, nh, act=tf.nn.relu, init_scale=1.0):
     with tf.variable_scope(scope):
        h = tf.layers.dense(x,nh,activattion=act,kernel_initializer = tf.orthogonal_initializer(init_scale))
        return h
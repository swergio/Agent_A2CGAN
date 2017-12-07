import numpy as np
import tensorflow as tf

from Model.Encoder.encoder import Encoder
from Model.Generator.generator import Generator
from Model.Critic.critic import Critic
from Model.Discriminator.discriminator import Discriminator

class offlineFakeModel(object):

    def __init__(self,latent_size, sess, env, batch_size,embedding_scope, reuse=False, init_X_msg = None, msg_length = None, initLenght = None, is_training=True):

        
        Z = tf.random_normal(shape=(batch_size, latent_size), mean=0., stddev=1.)
        #cty,cnr,mty,mes,act

        a,a_logits, mask = Generator(Z,env,batch_size,embedding_scope,reuse,init_X_msg, msg_length, initLenght)

        #v, v_logits  = Critic(e, reuse)

        d = Discriminator(a,Z,env,reuse)


        self.initial_state = np.zeros([batch_size,sum(env.message_space)*2])


        self.Z= Z
        self.a = a
        self.a_Logits = a_logits
        self.mask = mask
        #self.v = v
        #self.v_logits = v_logits
        self.d = d
        self.embedding_scope = embedding_scope
      
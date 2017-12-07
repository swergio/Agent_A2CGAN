import numpy as np

from Model.Encoder.encoder import Encoder
from Model.Generator.generator import Generator
from Model.Critic.critic import Critic
from Model.Discriminator.discriminator import Discriminator

class onlineModel(object):

    def __init__(self,Q,H, sess, env, batch_size,latent_size,  reuse=False, init_X_msg = None, msg_length = None, initLenght = None, is_training=True):

        e, e_stats, embedding_scope = Encoder(Q,H,latent_size, env, reuse, is_training = is_training)
        
        #cty,cnr,mty,mes,act

        a,a_logits, mask = Generator(e,env,batch_size,embedding_scope,reuse,init_X_msg, msg_length, initLenght)

        v, v_logits  = Critic(e, reuse)

        d = Discriminator(a,e,env,reuse)

        self.Q = Q
        self.H = H
        self.e = e
        self.e_stats = e_stats
        self.a = a
        self.a_Logits = a_logits
        self.mask = mask
        self.v = v
        self.v_logits = v_logits
        self.d = d
        self.embedding_scope = embedding_scope
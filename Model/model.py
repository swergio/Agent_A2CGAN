import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf


from Model.SubModels.onlineModel import onlineModel
from Model.SubModels.offlineRealModel import offlineRealModel
from Model.SubModels.offlineFakeModel import offlineFakeModel

from Model.Loss.lossFunctions import Policy_Loss, ValueFunction_Loss, Latent_Loss, GAN_Discriminator_Loss
from Model.Loss.helper import GAN_generator_feedback

from Model.Trainer.offlineTrainer import offlineTrainer
from Model.Trainer.onlineTrainer import onlineTrainer

from Model.Utility.ops import find_trainable_variables, make_path

import random



class Model(object):

    def __init__(self, env, 
            batch_size = 10, 
            memory_size = 10, 
            latent_size = 50,
            ent_coef=0.01,
            max_grad_norm=0.5,
            alpha=0.99, 
            epsilon=1e-5,
            latent_loss_weight = 1, 
            generation_loss_weight = 1,
            GAN_G_loss_weight = 1,
            GAN_D_loss_weight = 1,
            policy_loss_weight = 1,
            critic_loss_weight = 1):
        
        
        config = tf.ConfigProto()
      
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)

        self.batch_size = nbatch  = batch_size
        
        self.MemorySize = MemorySize = memory_size
        message_space =  env.message_space
        vocab_space = env.vocab_space
        MemNH = sum(message_space)

        self.alpha = alpha
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm

        self.latent_loss_weight =  latent_loss_weight
        self.generation_loss_weight = generation_loss_weight
        self.GAN_G_loss_weight = GAN_G_loss_weight
        self.GAN_D_loss_weight = GAN_D_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.critic_loss_weight = critic_loss_weight

        '''
        Declare Variables
        '''

        self.Q_step = Q_step = tf.placeholder(tf.int32, [1,MemNH]) #obs
        self.H_step = H_step = tf.placeholder(tf.int32,[1,MemorySize,MemNH])

        self.Init_msg_step  = Init_msg_step = tf.placeholder(tf.int32,shape = [1,message_space[3]])
        self.max_msg_l_step = max_msg_l_step = tf.placeholder(tf.int32,[1])
        self.init_msg_l_step = init_msg_l_step = tf.placeholder(tf.int32,[1])

        self.Q = Q = tf.placeholder(tf.int32, [nbatch,MemNH]) #obs
        self.H = H = tf.placeholder(tf.int32,[nbatch,MemorySize,MemNH])
        self.A = A = tf.placeholder(tf.int32, [nbatch,sum(message_space)])
        self.M = M =  tf.placeholder(tf.float32, [nbatch])

        self.ADV = ADV = tf.placeholder(tf.float32, [nbatch])
        self.R = R = tf.placeholder(tf.float32, [nbatch])

        self.Q_real = Q_real = tf.placeholder(tf.int32, [nbatch,MemNH]) #obs
        self.H_real = H_real = tf.placeholder(tf.int32,[nbatch,MemorySize,MemNH])
        self.A_real = A_real = tf.placeholder(tf.int32, [nbatch,sum(message_space)])
        self.M_real = M_real =  tf.placeholder(tf.float32, [nbatch])

        self.Init_msg = Init_msg = tf.placeholder(tf.int32,shape = [nbatch,message_space[3]])
        self.max_msg_l = max_msg_l = tf.placeholder(tf.int32,[nbatch])
        self.init_msg_l = init_msg_l = tf.placeholder(tf.int32,[nbatch])

        self.LR = LR = tf.placeholder(tf.float32, [])
        self.LR_encoder = LR_encoder = tf.placeholder(tf.float32, [])
        self.LR_generator = LR_generator = tf.placeholder(tf.float32, [])
        self.LR_discriminator = LR_discriminator = tf.placeholder(tf.float32, [])

        '''
        Sub Models
        '''
        self.online_step_model  = online_step_model = onlineModel(Q_step,H_step, sess, env, 1,latent_size, reuse=False, init_X_msg = Init_msg_step, msg_length = max_msg_l_step, initLenght = init_msg_l_step, is_training=False)
        self.online_model = online_model = onlineModel(Q,H, sess, env, batch_size, latent_size, reuse=True, init_X_msg = Init_msg, msg_length = max_msg_l, initLenght = init_msg_l, is_training=True)
        self.offline_real_model = offline_real_model = offlineRealModel(A_real,Q_real,H_real, sess, env, batch_size,latent_size,  reuse=True, init_X_msg = Init_msg, msg_length = max_msg_l, initLenght = init_msg_l, is_training=True)
        self.offline_fake_model = offline_fake_model = offlineFakeModel(latent_size, sess, env, batch_size,online_step_model.embedding_scope, reuse=True, init_X_msg = Init_msg, msg_length = max_msg_l, initLenght = init_msg_l, is_training=True)

        '''
        Loss Definitions
        '''
        online_latent_loss = latent_loss_weight * Latent_Loss(online_model)
        online_generation_loss = generation_loss_weight * Policy_Loss(offline_real_model,A_real,message_space,ent_coef,M, ADV = tf.constant(1.0,shape = [nbatch]))
        online_GAN_G_loss = GAN_G_loss_weight * Policy_Loss(online_model,A,message_space,ent_coef,M, ADV = GAN_generator_feedback(online_model) )
        online_GAN_D_loss = GAN_D_loss_weight * GAN_Discriminator_Loss(offline_real_model,online_model)
        online_policy_loss = policy_loss_weight * Policy_Loss(online_model,A,message_space,ent_coef,M, ADV = ADV )
        online_critic_loss = critic_loss_weight * ValueFunction_Loss(online_model,R)

        self.online_encoder_loss = online_encoder_loss = online_latent_loss + online_generation_loss
        self.online_generator_loss = online_generator_loss =  online_policy_loss  + online_GAN_G_loss + online_generation_loss
        self.online_discriminator_loss = online_discriminator_loss = online_GAN_D_loss
        self.online_critic_loss = online_critic_loss

        self.offline_latent_loss =  offline_latent_loss = latent_loss_weight * Latent_Loss(offline_real_model)
        self.offline_generation_loss = offline_generation_loss = generation_loss_weight * Policy_Loss(offline_real_model,A_real,message_space,ent_coef,M_real, ADV = tf.constant(1.0,shape = [nbatch]))
        self.offline_GAN_G_loss = offline_GAN_G_loss = GAN_G_loss_weight * Policy_Loss(offline_fake_model,offline_fake_model.a,message_space,ent_coef,M_real, ADV = GAN_generator_feedback(offline_fake_model) )
        self.offline_GAN_D_loss = offline_GAN_D_loss = GAN_D_loss_weight * GAN_Discriminator_Loss(offline_real_model,offline_fake_model)

        self.offline_encoder_loss = offline_encoder_loss = offline_latent_loss + offline_generation_loss
        self.offline_generator_loss = offline_generator_loss =  offline_generation_loss + offline_GAN_G_loss
        self.offline_discriminator_loss = offline_discriminator_loss = offline_GAN_D_loss

        '''
        Parameter
        '''
        self.encoder_params = encoder_params = find_trainable_variables("encoder") 
        self.generator_params = generator_params = find_trainable_variables("generator") 
        self.discriminator_params = discriminator_params = find_trainable_variables("discriminator") 
        self.critic_params = critic_params = find_trainable_variables("crtitc") 

        
        params = encoder_params + generator_params + discriminator_params + critic_params

        '''
        Trainer
        '''

        offline_trainer = offlineTrainer(self,nbatch, env)
        online_trainer =  onlineTrainer(self,nbatch, env)

        self.train = online_trainer.train
        self.trainOffline  = offline_trainer.train
        self.examplesOffline = offline_trainer.example


        def save(save_path, filename= 'saved.pkl'):
            path = save_path + "/" + filename
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, path)
            print('Model is saved')

        def load(load_path, filename= 'saved.pkl'):
            path = load_path  + "/" +  filename
            if osp.isfile(path):
                loaded_params = joblib.load(path)
                restores = []
                for p, loaded_p in zip(params, loaded_params):
                    restores.append(p.assign(loaded_p))
                ps = sess.run(restores)
                print('Model is loading')

        self.save = save
        self.load = load

        
        def step(ob,state, *_args, **_kwargs):
            max_msg_length_array = np.full([1],env.LengthOfMessageText,dtype=np.int32)
            sup_msg_length_array = np.full([1],1,dtype=np.int32)
            Init_msg_step_array = env.MessageText.PadIndex(env.MessageText.IndexSartMessage(1),env.LengthOfMessageText)

            action, value = sess.run([online_step_model.a, online_step_model.v], {Q_step:ob,H_step:state, Init_msg_step:Init_msg_step_array , max_msg_l_step:max_msg_length_array, init_msg_l_step: sup_msg_length_array })
            return action, value

        def value(ob,state, *_args, **_kwargs):
            max_msg_length_array = np.full([1],env.LengthOfMessageText,dtype=np.int32)
            sup_msg_length_array = np.full([1],1,dtype=np.int32)
            Init_msg_step_array = env.MessageText.PadIndex(env.MessageText.IndexSartMessage(1),env.LengthOfMessageText)
            return sess.run(online_step_model.v, {Q_step:ob,H_step:state, Init_msg_step: Init_msg_step_array, max_msg_l_step:max_msg_length_array, init_msg_l_step: sup_msg_length_array })

        self.step = step 
        self.value = value 
     
        tf.global_variables_initializer().run(session=sess)

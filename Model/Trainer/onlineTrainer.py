import numpy as np
import tensorflow as tf

from Model.Trainer.helper import RMSPropTrainer


class onlineTrainer(object):
     def __init__(self,model,nbatch, env):

        train_encoder_params = model.encoder_params
        train_generator_params  = model.generator_params 
        train_discriminator_params  = model.discriminator_params
        train_critic_params = model.critic_params

        encoder_solver, _ = RMSPropTrainer(model.online_encoder_loss, train_encoder_params, model.LR, model.alpha , model.epsilon, model.max_grad_norm, 'online_encoder_solver')
        generator_solver, _ = RMSPropTrainer(model.online_generator_loss, train_generator_params, model.LR, model.alpha , model.epsilon, model.max_grad_norm, 'online_generator_solver')
        discriminator_solver, _ = RMSPropTrainer(model.online_discriminator_loss, train_discriminator_params, model.LR, model.alpha , model.epsilon, model.max_grad_norm, 'online_discriminator_solver')
        critic_solver, _ = RMSPropTrainer(model.online_critic_loss, train_critic_params, model.LR, model.alpha , model.epsilon, model.max_grad_norm, 'online_critic_solver')
       

        def prepData(expertdata,textLength = env.LengthOfMessageText ,initTextLength = 1):
            Prior,Current ,Hist_Temp = expertdata.next_batch(nbatch)
            _ , mtxt_part_pad = env.InternalMessageHandler.MaskTextOfInternalMessage(Current,initTextLength, AddStartToken = True)
            Current, _ = env.InternalMessageHandler.MaskTextOfInternalMessage(Current, textLength)
            Hist = []
            for i,S in enumerate(Hist_Temp):
                if len(S) < model.MemorySize:
                    S_z = np.zeros((model.MemorySize- len(S),Current[i].shape[0]))
                    Hist.append(np.append(S_z,S).reshape((model.MemorySize,Current[i].shape[0])))
            Hist = np.asarray(Hist)
            if Prior == []:
                Prior = np.zeros(Current.shape)
            return Current, Prior, Hist , mtxt_part_pad


        def train(lr, obs, states, rewards, masks, actions, advantages, expertdata, textLength = env.LengthOfMessageText ,initTextLength = 1):
            
            max_msg_length_array = np.full([nbatch],textLength,dtype=np.int32)
            sup_msg_length_array = np.full([nbatch],initTextLength,dtype=np.int32)
            m_array = np.full([nbatch],1, dtype = np.float32)

            Current, Prior, Hist , mtxt_part_pad  = prepData(expertdata,textLength,initTextLength) 

            encoder_loss, generator_loss, discriminator_loss,critic_loss, _, _, _,_ = model.sess.run(
                [
                    model.online_encoder_loss,
                    model.online_generator_loss,
                    model.online_discriminator_loss,
                    model.online_critic_loss,
                    encoder_solver, 
                    generator_solver,
                    discriminator_solver,
                    critic_solver
                ],
                feed_dict={
                    model.Q : obs,
                    model.H : states,
                    model.A : actions,
                    model.M : masks,
                    model.R : rewards,
                    model.ADV : advantages,
                    model.Q_real : Prior,
                    model.H_real : Hist,
                    model.A_real : Current,
                    model.M_real : m_array,
                    model.Init_msg : mtxt_part_pad,
                    model.max_msg_l : max_msg_length_array,
                    model.init_msg_l : sup_msg_length_array,
                    model.LR : lr
                }
            )

            summary = tf.Summary()
            mydata = {"encoder_loss": encoder_loss, 
                        "generator_loss": generator_loss, 
                        "discriminator_loss": discriminator_loss, 
                        "critic_loss": critic_loss}
            for name, data in mydata.items():
                summary.value.add(tag=name, simple_value=data)

            return encoder_loss, generator_loss, discriminator_loss,critic_loss, summary

        self.train = train

 

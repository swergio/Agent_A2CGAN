import numpy as np
import tensorflow as tf

from Model.Trainer.helper import RMSPropTrainer


class offlineTrainer(object):
    def __init__(self,model,nbatch, env):



        train_encoder_params = model.encoder_params
        train_generator_params  = model.generator_params
        train_discriminator_params  = model.discriminator_params

        encoder_solver,_ = RMSPropTrainer(model.offline_encoder_loss, train_encoder_params, model.LR_encoder, model.alpha , model.epsilon, model.max_grad_norm, 'offline_encoder_solver')
        generator_solver,_ = RMSPropTrainer(model.offline_generator_loss, train_generator_params, model.LR_generator, model.alpha , model.epsilon, model.max_grad_norm, 'offline_generator_solver')
        discriminator_solver,_ = RMSPropTrainer(model.offline_discriminator_loss, train_discriminator_params, model.LR_discriminator, model.alpha , model.epsilon, model.max_grad_norm, 'offline_discriminator_solver')
       

        def prepData(expertdata,textLength = env.LengthOfMessageText ,initTextLength = 1):
            Prior,Current ,Hist_Temp = expertdata.next_batch(nbatch)
            _ , mtxt_part_pad = env.InternalMessageHandler.MaskTextOfInternalMessage(Current,initTextLength, AddStartToken = True)
            Current, _ = env.InternalMessageHandler.MaskTextOfInternalMessage(Current, textLength)

            Hist = []
            for S in Hist_Temp:
                if len(S) < model.MemorySize:
                    S_z = np.zeros((model.MemorySize- len(S),Current[0].shape[0]))
                    Hist.append(np.append(S_z,S).reshape((model.MemorySize,Current[0].shape[0])))
            Hist = np.asarray(Hist)

            if Prior == []:
                Prior = np.zeros(Current.shape)

            return Current, Prior, Hist , mtxt_part_pad


        def train(encoder_lr, generator_lr, discriminator_lr, expertdata,textLength = env.LengthOfMessageText ,initTextLength = 1):
            max_msg_length_array = np.full([nbatch],textLength,dtype=np.int32)
            sup_msg_length_array = np.full([nbatch],initTextLength,dtype=np.int32)
            m_array = np.full([nbatch],1, dtype = np.float32)

            Current, Prior, Hist , mtxt_part_pad  = prepData(expertdata,textLength,initTextLength) 

            encoder_loss, generator_loss, discriminator_loss, _, _, _ , latent_loss, generation_loss, GAN_G_loss, GAN_D_loss, d_real, d_fake = model.sess.run(
                [
                    model.offline_encoder_loss,
                    model.offline_generator_loss,
                    model.offline_discriminator_loss,
                    encoder_solver, 
                    generator_solver,
                    discriminator_solver,
                    model.offline_latent_loss,
                    model.offline_generation_loss, 
                    model.offline_GAN_G_loss,
                    model.offline_GAN_D_loss,
                    model.offline_real_model.d,
                    model.offline_fake_model.d
                ],
                feed_dict={
                    model.Q_real : Prior,
                    model.H_real : Hist,
                    model.A_real : Current,
                    model.M_real : m_array,
                    model.Init_msg : mtxt_part_pad,
                    model.max_msg_l : max_msg_length_array,
                    model.init_msg_l : sup_msg_length_array,
                    model.LR_encoder : encoder_lr,
                    model.LR_generator : generator_lr,
                    model.LR_discriminator : discriminator_lr
                }
            )

            '''
            Train GAN Discriminator again
            '''
            for _ in range(2):
                Current, Prior, Hist , mtxt_part_pad  = prepData(expertdata,textLength,initTextLength) 

                discriminator_loss, _ ,GAN_D_loss = model.sess.run(
                    [
                        model.offline_discriminator_loss,
                        discriminator_solver,
                        model.offline_GAN_D_loss
                    ],
                    feed_dict={
                        model.Q_real : Prior,
                        model.H_real : Hist,
                        model.A_real : Current,
                        model.M_real : m_array,
                        model.Init_msg : mtxt_part_pad,
                        model.max_msg_l : max_msg_length_array,
                        model.init_msg_l : sup_msg_length_array,
                        model.LR_discriminator : discriminator_lr 
                    }
                )

            
            summary = tf.Summary()
            mydata = {"encoder_loss": encoder_loss, 
                "generator_loss": generator_loss,
                "discriminator_loss": discriminator_loss,
                "latent_loss": latent_loss,
                "generation_loss": generation_loss,
                "GAN_G_loss": GAN_G_loss,
                "GAN_D_loss": GAN_D_loss}
            for name, data in mydata.items():
                summary.value.add(tag=name, simple_value=data)
           


            return encoder_loss, generator_loss, discriminator_loss, summary, d_real, d_fake


        def printResults(PrintN, Current, Prior, Hist, Latent_Real, Generated_Current, Sampled_Current, Sampled_Discriminator):
            n = min(PrintN,nbatch)

            for i in range(n):
                cur = env.InternalMessageHandler.InternalToExternalMessage([Current[i]])
                pri = env.InternalMessageHandler.InternalToExternalMessage([Prior[i]])
                #his = env.InternalMessageHandler.InternalToExternalMessage([Hist[i]])
                gen_cur = env.InternalMessageHandler.InternalToExternalMessage([Generated_Current[i]])
                sam_cur = env.InternalMessageHandler.InternalToExternalMessage([Sampled_Current[i]])
                
                def printMes(description, mes):
                    print(description + ' from ' +  mes.NameSpace + ' as '+  mes.MessageType + ': ' +  mes.Data )

                print('EXAMPLE ' + str(i) + ':')
                printMes('Prior Observation',pri)
                printMes('Real Action',cur)
                printMes('Generated Action',gen_cur)
                printMes('Sampled Action',sam_cur)
                print('Descrimitator:' + str(Sampled_Discriminator[i]))
                #print('Real Latent Vektor:' + str(Latent_Real[i]))

        def example(expertdata,textLength = env.LengthOfMessageText,initTextLength = 1, printNresults = 0):
            max_msg_length_array = np.full([nbatch],textLength,dtype=np.int32)
            sup_msg_length_array = np.full([nbatch],initTextLength,dtype=np.int32)
            m_array = np.full([nbatch],1, dtype = np.int32)

            Current, Prior, Hist , mtxt_part_pad  = prepData(expertdata,textLength,initTextLength) 

            Latent_Real, Generated_Current, Sampled_Current, Sampled_Discriminator, latent_stats= model.sess.run(
                [
                    model.offline_real_model.e,
                    model.offline_real_model.a,
                    model.offline_fake_model.a,
                    model.offline_fake_model.d,
                    model.offline_real_model.e_stats
                ],
                feed_dict={
                    model.Q_real : Prior,
                    model.H_real : Hist,
                    model.A_real : Current,
                    model.M_real : m_array,
                    model.Init_msg : mtxt_part_pad,
                    model.max_msg_l : max_msg_length_array,
                    model.init_msg_l : sup_msg_length_array
                }
            ) 

            if printNresults > 0: 
                printResults(printNresults, Current, Prior, Hist, Latent_Real, Generated_Current, Sampled_Current, Sampled_Discriminator )

            return Current, Prior, Hist, Latent_Real, Generated_Current, Sampled_Current, Sampled_Discriminator


                
        self.train = train 
        self.example = example

    

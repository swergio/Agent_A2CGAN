import tensorflow as tf
import numpy as np 
import time

class Learner(object):

    def __init__(self, Runner):
        self.Runner = Runner
        self.lr = Runner.lr

        self.update = 1
        self.update_O_GAN = 1
        self.update_AO_GAN = 1
        self.train_writer = tf.summary.FileWriter(self.Runner.logpath)

        self.ActionStats = self.ResetActionStats()
        self.sumOfRewards = 0
        self.log_interval= self.Runner.log_interval
        self.save_interval = self.Runner.save_interval


    def learn(self, obs, states, rewards, masks, actions, values,advantages):
            update = self.update

            encoder_loss, generator_loss, discriminator_loss,critic_loss, summary = self.Runner.model.train(self.lr,obs, states, rewards, masks, actions,advantages, self.Runner.expertdata)
            nbatch = self.Runner.nsteps 
            tstart = time.time()
            nseconds = time.time()-tstart
            self.train_writer.add_summary(summary,update)

            def printlog(key,val, key_width,val_width):
                
                print('| %s%s | %s%s |' % (
                    key,
                    ' ' * (key_width - len(key)),
                    str(val),
                    ' ' * (val_width - len(val)),
                ))

            key_width = 25
            val_width = 15
            if update % self.log_interval == 0:
                actStat = self.ActionStats
                # Print values
                print('-' * (key_width + val_width + 7))
                printlog("action_CTYP_AGENT",actStat['ChatType'][0],key_width,val_width)
                printlog("action_CTYP_WORKER",actStat['ChatType'][1],key_width,val_width)
                printlog("action_CTYP_KNOWLEDGE",actStat['ChatType'][2],key_width,val_width)
                printlog("action_CNR_0",actStat['ChatNr'][0],key_width,val_width)
                printlog("action_CNR_1",actStat['ChatNr'][1],key_width,val_width)
                printlog("action_MTYP_QUESTION",actStat['MessageType'][0],key_width,val_width)
                printlog("action_MTYP_ANSWER",actStat['MessageType'][1],key_width,val_width)
                printlog("nupdates", update,key_width,val_width)
                printlog("avg_reward",self.sumOfRewards/(nbatch*self.log_interval),key_width,val_width)
                printlog("sum_reward",self.sumOfRewards,key_width,val_width)
                printlog("total_timesteps", update*nbatch,key_width,val_width)
                printlog("encoder_loss", float(encoder_loss),key_width,val_width)
                printlog("generator_loss", float(generator_loss),key_width,val_width)
                printlog("discriminator_loss", float(discriminator_loss),key_width,val_width)
                printlog("critic_loss", float(critic_loss),key_width,val_width)
                print('-' * (key_width + val_width + 7))
                logger.dump_tabular()
                self.sumOfRewards = 0
                self.ResetActionStats()

            if update % 10 == 0:
                print(str(update) +'th update')

            if update % self.save_interval == 0:
                self.Runner.model.save(self.Runner.modelsavepath)
            
            self.update = update+ 1


    def ResetActionStats(self):
        action_space = self.Runner.env.action_space
        ncty = action_space[0]
        ncnr = action_space [1]
        nmty = action_space [2]

        dic = {}  
        dic['ChatType'] = np.zeros(ncty)
        dic['ChatNr'] = np.zeros(ncnr)
        dic['MessageType'] = np.zeros(nmty)
        dic['Acted'] = np.zeros(1)
        dic['NotActed'] = np.zeros(1)
        self.ActionStats = dic
        return dic

    def actionStats(self, actions):
        ctyi, cnri, mtyi, mtxt, act = np.split(actions,np.cumsum(self.Runner.env.message_space), axis=1)[:-1]
        ctyi = ctyi[0][0]
        cnri = cnri[0][0]
        mtyi = mtyi[0][0]
        act = act[0][0]
        if act == 1:
            self.ActionStats['Acted'][0] += 1
            self.ActionStats['ChatType'][ctyi] += 1
            self.ActionStats['ChatNr'][cnri] += 1
            self.ActionStats['MessageType'][mtyi] += 1
        else:
            self.ActionStats['NotActed'][0] += 1

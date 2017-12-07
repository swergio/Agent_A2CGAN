import os.path as osp
import time
import joblib
import logging
import numpy as np
import tensorflow as tf

from Runner.Utility.functions import sigmoid



import random

class Runner(object):

    def __init__(self, env, model,modelsavepath, logpath, lr = 7e-4,  train_steps=10000,log_interval = 500, save_interval = 2000):
        
        self.env = env
        self.model = model
        self.nsteps = nsteps = model.batch_size
        message_space = self.env.message_space
        nh = sum(message_space)
        self.batch_ob_shape = (nsteps, nh)
        self.initObs = np.zeros((1, nh), dtype=np.uint8)
        self.StateMemory = model.MemorySize
        self.initMemoryState = np.zeros((1,self.StateMemory,nh))
        self.initDones = [False for _ in range(1)]
        self.lr = lr
        

        self.train_steps = train_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
       
        self.step = 1
 
        self.modelsavepath = modelsavepath

        self.expertdata = env.ExpertData
        self.expertdata.loadData()
        
        self.firstTime = True
        
        
        # Load Model params if save exist
        try:
            self.model.load(self.modelsavepath)
            print('Load Model.....')
        except:
            print('Can not load model')
     
        self.train_writer = tf.summary.FileWriter(logpath) #, self.model.sess.graph)
        

    
    def train(self):

        supportl = 1
        msgl = 1
        supmsgldiff = 0

        d_real = 0.5
        d_fake = 0.5


        self.expertdata.UpdateSampleSettings( Types = ['A'])

        maxSup = self.expertdata.GetmaxTxtLength()
        maxDeltaStep = 1
        maxLen = maxSup +maxDeltaStep
        print('MaxLength :' + str(maxLen))
        minSteps = (maxLen*(maxLen+1)/2) - maxDeltaStep*(maxDeltaStep+1)/2 

        changeSupportEach = int(self.train_steps / minSteps)

        print('Start training...')
        print('support : ' + str(supportl) + ' msgl : ' + str(msgl))
        for i in range(self.train_steps):
            
            if i % changeSupportEach == 0 and i != 0:
                if  msgl + 1 <=  maxLen and supportl + 1 <= maxSup:
                    supportl = supportl + 1
                else:
                    supportl  = 1
                    supmsgldiff = supmsgldiff + 1
                msgl = supportl + supmsgldiff 
                print('support : ' + str(supportl) + ' msgl : ' + str(msgl))
            
            encoder_lr, generator_lr, discriminator_lr = self.adjustedLR(d_real, d_fake)

            encoder_loss, generator_loss, discriminator_loss, summary,  d_real, d_fake= self.model.trainOffline(encoder_lr, generator_lr, discriminator_lr,self.expertdata,textLength = msgl ,initTextLength = supportl )
            self.train_writer.add_summary(summary,self.step)
            self.step += 1
            if i % 100 == 0:
                print('Step: ' + str(i))
              

            if i % self.log_interval == 0 and i != 0:
                self.model.examplesOffline(self.expertdata,textLength = msgl,initTextLength = supportl, printNresults = 2)
                print ('dreal: ' + str(np.mean(d_real)) + ' dfake: ' + str(np.mean(d_fake)) )
                encoder_lr, generator_lr, discriminator_lr = self.adjustedLR(d_real, d_fake)
                print('encoder_lr: ' + str(encoder_lr) + ' generator_lr: ' + str(generator_lr) + ' discriminator_lr: ' + str(discriminator_lr) )

            if i %self.save_interval == 0 and i != 0:
                self.model.save(self.modelsavepath)

        self.model.save(self.modelsavepath)

    def adjustedLR(self,d_real, d_fake):
        encoder_lr = self.lr *sigmoid(np.mean(np.clip(d_real,1e-5,1)),-.5,15)
        generator_lr = self.lr *sigmoid(np.mean(np.clip(d_real,1e-5,1)),-.5,15)
        discriminator_lr = self.lr *sigmoid(np.mean(np.clip(d_fake,1e-5,1)) + (1- np.mean(np.clip(d_real,1e-5,1))),-.5,15)

        return encoder_lr, generator_lr, discriminator_lr


        

   




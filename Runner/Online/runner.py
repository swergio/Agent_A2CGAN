import os.path as osp
import time
import logging
import numpy as np
import tensorflow as tf

from Runner.Online.actor import Actor
from Runner.Online.learner import Learner


import random

class Runner(object):

    def __init__(self, env, model,modelsavepath, logpath,
        lr = 7e-4,
        gamma=0.99,
        gae_lambda =  0.96,
        log_interval = 100, 
        save_interval = 500,
        act_probability = 1,
        explor_probability = 0.05):
        
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
        
        self.modelsavepath = modelsavepath
        self.logpath = logpath
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.act_probability = act_probability
        self.explor_probability = explor_probability

       
       
        

        self.xps = []
        self.actor = actor =  Actor(self)
        self.learner = learner = Learner(self)
        
        

        self.expertdata = env.ExpertData

        self.expertdata.loadData()
        self.expertdata.UpdateSampleSettings(Types = ['A'])
        
        self.firstTime = True
        
        
        # Load Model params if save exist
        try:
            self.model.load(self.modelsavepath)
            print('Load Model.....')
        except:
            print('Can not load model')
        
        
        self.env.SetCallbackFunction(actor.act)

        self.learn_cb = learner.learn
    
    def listen(self):
        self.env.ListenToSocketIO()
        

   




import random 
import numpy as np

from Runner.Online.experience import Experience

class Actor(object):

    def __init__(self, Runner):
        self.Runner = Runner
        self.obs = np.copy(Runner.initObs)
        self.states = np.copy(self.Runner.initMemoryState)
        self.dones = np.copy(self.Runner.initDones)


    def act(self,observation, reward, done, comID, step,CommunicationID,Namespace,MessageType):
            #Update  Runner.observation
            observation = observation.reshape(self.obs.shape)      
            self.obs = observation
            #set reward and done to arrays
            self.Runner.learner.sumOfRewards =+ reward
            reward = [reward]
            done = [done]
            self.dones = done

            #if first message for communication create new expierence class
            # if not first step append reward and done
            if step == 1:
                xp = Experience(comID, self.Runner.gamma, self.Runner.gae_lambda, self.Runner.nsteps,self.states)
                self.Runner.xps.append(xp)
            else:
                xp = [m for m in self.Runner.xps if m.ID == comID][0]
                xp.xp_rewards.append(reward)

            #Add Done to xp => dones length +1 then other
            xp.xp_dones.append(self.dones)

            #decide if to train model, call callback train function
            # if response to nth action or if done
            if (step != 1 and step % self.Runner.nsteps == 1) or (done[0] and len(xp.xp_obs) > 0):
                #transofrom arrays and call learn callback function
                lastValues = self.Runner.model.value(self.obs, self.states, self.dones).tolist()

                xp_obs, xp_states, xp_rewards, xp_masks, xp_actions, xp_values, xp_advantages = xp.transform(self.states, self.Runner.batch_ob_shape,lastValues)
                
                self.Runner.learn_cb(xp_obs, xp_states, xp_rewards, xp_masks, xp_actions, xp_values,xp_advantages)
                #reset xp class
                xp.reset()
                #Add last done statement back to xp class
                xp.xp_dones.append(self.dones)

            actions, values = self.Runner.model.step(self.obs, self.states, self.dones)

            actions = self.exploration(actions)

            '''
            if last act is >0.x then act else set all to 0 and don't act
            '''
            DoAct = True
            actProp = actions[0][len(actions[0])-1]

            if min(max(actProp,0),1) > self.Runner.act_probability or MessageType == self.Runner.env.MessageType.values.FEEDBACK or done[0]:
                DoAct = False
                actions = np.zeros(actions.shape,dtype=np.int)
            else:
                actions[0][len(actions[0])-1] = 1.0

            xp.xp_obs.append(np.copy(self.obs))
            xp.xp_actions.append(actions)
            xp.xp_values.append(values)
            xp.xp_hist.append(np.copy(self.obs).tolist()[0])
            xp.xp_hist.append(np.copy(actions).tolist()[0])

            states = np.asarray(xp.xp_hist[max(0,len(xp.xp_hist)- self.Runner.StateMemory):max(0,len(xp.xp_hist))])
            states = np.expand_dims(states,axis = 0)
            xp.xp_states.append(np.copy(self.states))
            self.states = states
        
            self.Runner.learner.actionStats(actions)

            #Submit value to store in
            self.Runner.env.QuestionFeedback.OnIn(CommunicationID,Namespace,MessageType, values[0])

            if done[0]:
                self.Runner.xps.remove(xp)    
                self.states = np.copy(self.Runner.initMemoryState)
                self.dones = np.copy(self.Runner.initDones)
                self.Runner.env.QuestionFeedback.ResetQestionInOut()

            #call action
            if DoAct:
                self.Runner.env.emit(comID,actions, values)

    def exploration(self,actions):
        if random.random() > self.Runner.explor_probability:
            ret =  actions
        else:
            action_space = self.Runner.env.action_space
            ncty = action_space[0]
            ncnr = action_space[1]
            nmty = action_space[2]
            nmsg = action_space[3]

            message_space =  self.Runner.env.message_space

            cty = [random.sample(range(ncty),1)[0] for _ in range(message_space[0])]
            cnr = [random.sample(range(ncnr),1)[0] for _ in range(message_space[1])]
            mty = [random.sample(range(nmty),1)[0] for _ in range(message_space[2])]
            msg = [random.sample(range(nmsg),1)[0] for _ in range(message_space[3])]
            act = [1 if random.random() >= 0 else 0]
            
            ret = np.reshape(np.concatenate((cty,cnr,mty,msg,act)),(1,int(np.sum(message_space))))
        return ret
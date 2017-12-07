import numpy as np
from Runner.Utility.functions import discount_with_dones

class Experience():
    def __init__(self,id, gamma, gae_lambda, nsteps, histInit):
        self.ID = id
        self.gamma = gamma
        self.nsteps  = nsteps
        self.gae_lambda = gae_lambda
        self.xp_obs, self.xp_rewards, self.xp_actions, self.xp_values, self.xp_dones, self.xp_states = [],[],[],[],[], []
        xp_hist = []
        for i in range(len(histInit[0])):
            xp_hist.append(histInit[0][i].tolist())
        self.xp_hist = xp_hist

    def reset(self):
        self.xp_obs, self.xp_rewards, self.xp_actions, self.xp_values, self.xp_dones, self.xp_states = [],[],[],[],[], []

    def transform(self, xp_states, batch_ob_shape, last_values):
        if len(self.xp_obs) < self.nsteps:
            short = True
        else:
            short = False
        xp_obs = self.xp_obs
        xp_rewards = self.xp_rewards
        xp_actions = self.xp_actions
        xp_values = self.xp_values
        xp_dones = self.xp_dones
        xp_states = self.xp_states

        xp_masks = []
        [xp_masks.append([1]) for i in range(len(xp_obs))]
        #append missing nsteps
        if short:
            [xp_masks.append([0]) for i in range(self.nsteps-len(xp_masks))]
            [xp_obs.append(np.zeros(xp_obs[0].shape)) for i in range(self.nsteps-len(xp_obs))]
            [xp_rewards.append([0]) for i in range(self.nsteps-len(xp_rewards))]
            [xp_actions.append(np.zeros(xp_actions[0].shape)) for i in range(self.nsteps-len(xp_actions))]
            [xp_values.append(np.zeros(xp_values[0].shape)) for i in range(self.nsteps-len(xp_values))]
            [xp_dones.append([True]) for i in range(self.nsteps +1-len(xp_dones))]
            [xp_states.append(np.zeros(xp_states[0].shape)) for i in range(self.nsteps-len(xp_states))]


        #batch of steps to batch of rollouts
        xp_obs = np.asarray(xp_obs, dtype=np.uint8).swapaxes(1, 0).reshape(batch_ob_shape)
        xp_rewards = np.asarray(xp_rewards, dtype=np.float32).swapaxes(1, 0)
        xp_actions = np.asarray(xp_actions, dtype=np.int32).swapaxes(1, 0)
        xp_values = np.asarray(xp_values, dtype=np.float32).swapaxes(1, 0)
        xp_dones = np.asarray(xp_dones, dtype=np.bool).swapaxes(1, 0)
        xp_masks = np.asarray(xp_masks, dtype=np.float32).swapaxes(1, 0)
        xp_dones = xp_dones[:, 1:]
        
        xp_advantages = self.GenrelaziedAdvantageEstimate(xp_rewards,xp_values,xp_dones,last_values)

        for n, (rewards, dones, value) in enumerate(zip(xp_rewards, xp_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1] #Don't get it why not use last?
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            xp_rewards[n] = rewards
        xp_rewards = np.squeeze(xp_rewards, axis=0)
        xp_advantages = np.squeeze(xp_advantages, axis=0)
        xp_actions = np.squeeze(xp_actions, axis=0)
        xp_values = np.squeeze(xp_values, axis=0)
        xp_masks = np.squeeze(xp_masks, axis=0)
        xp_states = np.squeeze(xp_states, axis=0)
       
        return xp_obs, xp_states, xp_rewards, xp_masks, xp_actions, xp_values, xp_advantages

    def GenrelaziedAdvantageEstimate(self,xp_rewards,xp_values,xp_dones,last_values ):
        gamma = self.gamma
        gae_lambda = self.gae_lambda
        xp_advantages = np.zeros(xp_rewards.shape)


        for n, (rewards,values,dones, lastvalue) in enumerate(zip(xp_rewards,xp_values, xp_dones, last_values)):
            rewards = rewards.tolist()
            values = values.tolist()
            dones = dones.tolist()
            values = values + [lastvalue]
            adv = np.asarray(rewards) + gamma*np.asarray(values[1:])  - np.asarray(values[:-1])
            advantages = discount_with_dones(adv.tolist(), dones, gamma * gae_lambda)

            xp_advantages[n] = advantages

        return xp_advantages




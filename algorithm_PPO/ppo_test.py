import numpy as np
import os
import scipy.io
import torch

import my_env
from ppo_algo import PPO
from ppo_algo import Memory
import matplotlib.pyplot as plt

IS_TEST = 1

# ################## SETTINGS ######################

lane_num = 3
n_veh = 9
width = 120

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# ------- characteristics related to the network -------- #
env = my_env.Environment(lane_num, n_veh, width)
env.make_new_game() # initialize parameters in env

n_step_per_episode = 100
n_episode_test = 10  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(env.get_state())
n_output = 2 * n_veh  # channel selection, power, phase

update_timestep = 100  # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 20#80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr = 0.0001  # parameters for Adam optimizer
betas = (0.9, 0.999)

# --------------------------------------------------------------
memory = Memory()
agent = PPO(n_input, n_output, action_std, lr, betas, gamma, K_epochs, eps_clip)

label = 'model/ppo_model'
model_path = label + '/n_%d'%n_veh

agent.load_model(model_path)

## Let's go
if IS_TEST:
    # agent.load_models()
    record_reward_average = []
    t_dt = []
    t_tk = []
    alloc_tk = []
    alloc_dt = []
    time_step = 0
    for i_episode in range(n_episode_test):
        done = False
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)

        if i_episode % 20 == 0:
            env.vehicle_renew_position()
            env.renew_channel()
            env.R_V2I()

        state_old_all = []
        #for i in range(n_veh):
        state = env.get_state()
        state_old_all.append(state)

        average_reward = 0
        time_dt = 0
        time_tk = 0
        allocation_tk = 0
        allocation_dt = 0
        for i_step in range(n_step_per_episode):
            time_step += 1
            state_new_all = []
            action_all_training = np.zeros([n_veh, 2], dtype=np.float16)  # sub, power
            # receive observation
            action = agent.select_action(np.asarray(state_old_all).flatten(), memory)
            action = np.clip(action, 0.01, 0.22)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_veh):
                action_all_training[i, 0] = action[0+i*2]/2*100  # chosen RB
                action_all_training[i, 1] = action[1+i*2]/2*100  # power selected by PL
            action_channel = action_all_training.copy()
            _, train_reward, done = env.act_for_training(action_channel)
            time_dt += np.sum(env.t_DT)/n_veh
            time_tk += np.sum(env.t_TK)/n_veh
            allocation_tk +=np.sum(action_channel,axis=0)[0]
            allocation_dt +=np.sum(action_channel,axis=0)[1]

            record_reward[i_step] = train_reward

            # get new state
            #for i in range(n_veh):
            state_new = env.get_state()
            state_new_all.append((state_new))

            state_old_all = state_new_all

        time_dt = time_dt/n_step_per_episode
        time_tk = time_tk/n_step_per_episode
        alloc_tk.append(allocation_tk/n_step_per_episode)
        alloc_dt.append(allocation_dt/n_step_per_episode)
        t_dt.append(time_dt)
        t_tk.append(time_tk)
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
    print('average returns:', np.mean(record_reward_average))
    print('\nTime consumption for maintaining digital twins:', sum(t_dt)/n_episode_test)
    print('\nTime consumption for computing tasks:', sum(t_tk)/n_episode_test)
    print('\nResources for maintaining digital twins:', sum(alloc_dt)/n_episode_test)
    print('\nResources for computation tasks:', sum(alloc_tk)/n_episode_test)

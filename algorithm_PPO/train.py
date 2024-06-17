import numpy as np
import os
import scipy.io
import torch

import my_env
from ppo_algo import PPO
from ppo_algo import Memory
import matplotlib.pyplot as plt

IS_TRAIN = 1

# ################## SETTINGS ######################

lane_num = 3
n_veh = 5
width = 120

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# ------- characteristics related to the network -------- #
env = my_env.Environment(lane_num, n_veh)#, width)
env.make_new_game() # initialize parameters in env

n_episode = 500
n_step_per_episode = 100
n_episode_test = 100  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(env.get_state())
n_output = 2 * n_veh  # channel selection, power, phase

update_timestep = 100  # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 20#80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.95  # discount factor

lr = 0.0001  # parameters for Adam optimizer
betas = (0.9, 0.999)

# --------------------------------------------------------------
memory = Memory()
agent = PPO(n_input, n_output, action_std, lr, betas, gamma, K_epochs, eps_clip)

label = 'model/ppo_model'
model_path = label + '/n_%d'%n_veh

## Let's go
if IS_TRAIN:
    # agent.load_models()
    record_reward_average = []
    time_step = 0
    for i_episode in range(n_episode):
        done = False
        print("-------------------------------------------------------------------------------------------------------")
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
        for i_step in range(n_step_per_episode):
            time_step += 1
            state_new_all = []
            action_all_training = np.zeros([n_veh, 2], dtype=np.float16)  # sub, power
            # receive observation
            action = agent.select_action(np.asarray(state_old_all).flatten(), memory)
            action = np.clip(action, 0.01, 0.3)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_veh):
                action_all_training[i, 0] = action[0+i*2]/2*100  # chosen RB
                action_all_training[i, 1] = action[1+i*2]/2*100  # power selected by PL
            action_channel = action_all_training.copy()
            _, train_reward, done = env.act_for_training(action_channel)

            record_reward[i_step] = train_reward

            # get new state
            #for i in range(n_veh):
            state_new = env.get_state()
            state_new_all.append((state_new))

            state_old_all = state_new_all

            memory.rewards.append(train_reward)
            memory.is_terminals.append(done)


            # update if its time
            if time_step % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                time_step = 0

        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
        print('step:', i_episode, 'reward', average_reward)

        if (i_episode+1) % 100 == 0 and i_episode != 0:
            agent.save_model(model_path)

    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average
    np.save(model_path + 'returns.pkl', record_reward_average)
    plt.figure(2)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    print('Training Done. Saving models...')
    '''current_dir = os.path.dirname(os.path.realpath(__file__))

    reward_path = os.path.join(current_dir, "model/" + label + '/reward.mat')

    scipy.io.savemat(reward_path, {'reward': record_reward_average})'''


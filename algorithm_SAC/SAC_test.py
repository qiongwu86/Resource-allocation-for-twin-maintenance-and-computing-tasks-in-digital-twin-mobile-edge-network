import numpy as np
import os
import scipy.io
import my_env_sac
from RL_SAC import SAC_Trainer
from  RL_SAC import ReplayBuffer
import matplotlib.pyplot as plt

########################SETTING######################
lane_num = 3
n_veh = 4
width = 120

IS_TEST = 1

label = 'model/sac_model/n_%d'%n_veh
model_path = label + '/agent'

env = my_env_sac.Environment(lane_num, n_veh, width)
env.make_new_game()

n_step_per_episode = 100
n_episode_test = 50  # test episodes

#####################################################

def get_State(env, ind_episode=1., epsi=0.02):
    D_tk = env.task_size
    D_dt = env.upd_size
    D_tk_fc = env.task_fc
    D_dt_fc = env.upd_fc
    D_tk_limit = env.tk_t_limit
    D_dt_limit = env.upd_t_limit
    vehicle_v = env.ve_v
    vehicle_l = env.ve_l
    vehicle_G = env.g_channel
    return np.concatenate((D_tk, D_tk_fc,D_tk_limit, D_dt, D_dt_fc,D_dt_limit,vehicle_v, vehicle_l, vehicle_G, [ind_episode, epsi]))
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# ------- characteristics related to the network -------- #
batch_size = 64
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(get_State(env=env))
n_output = 2*n_veh
action_range = 1.0
# --------------------------------------------------------------
#agent = SAC_Trainer(alpha, beta, n_input, tau, gamma, 12 ,memory_size, fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, 2, 'OU')
replay_buffer_size = 2e5#1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 512
agent = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)
agent.load_model(model_path)
DETERMINISTIC=False
## Let's go
if IS_TEST:
    record_reward_average = []
    t_dt = []
    t_tk = []
    alloc_tk = []
    alloc_dt = []
    print("\nRestoring the sac model...")
    for i_episode in range(n_episode_test):
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)

        #env.make_new_game()
        if i_episode % 20 == 0:
            env.vehicle_renew_position() # update vehicle position
            env.renew_channel()
            env.R_V2I()


        state_old_all = []
        #for i in range(n_veh):
        state = get_State(env)
        state_old_all.append(state)

        average_reward = 0
        time_dt = 0
        time_tk = 0
        allocation_tk = 0
        allocation_dt = 0
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, 2], dtype=np.float16)  # sub, power
            # receive observation
            action = agent.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            action = np.clip(action, 0.01, 0.27)
            action_all.append(action)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_veh):
                action_all_training[i, 0] = ((action[0+i*2])/2 ) * 100  # chosen RB
                action_all_training[i, 1] = ((action[1+i*2])/2 ) * 100
            action_channel = action_all_training.copy()
            train_reward = env.act_for_training(action_channel)
            time_dt += np.sum(env.t_DT)/n_veh
            time_tk += np.sum(env.t_TK)/n_veh
            allocation_tk +=np.sum(action_channel,axis=0)[0]
            allocation_dt +=np.sum(action_channel,axis=0)[1]

            record_reward[i_step] = train_reward

            # get new state
            #for i in range(n_veh):
            state_new = get_State(env)
            state_new_all.append((state_new))
            # old observation = new_observation
            state_old_all = state_new_all
        time_dt = time_dt/n_step_per_episode
        time_tk = time_tk/n_step_per_episode
        alloc_tk.append(allocation_tk/n_step_per_episode)
        alloc_dt.append(allocation_dt/n_step_per_episode)
        t_dt.append(time_dt)
        t_tk.append(time_tk)
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
    print(action_channel)
    print('average returns:', np.mean(record_reward_average))
    print('\nTime consumption for maintaining digital twins:', sum(t_dt)/n_episode_test)
    print('\nTime consumption for computing tasks:', sum(t_tk)/n_episode_test)
    print('\nResources for maintaining digital twins:', sum(alloc_dt)/n_episode_test)
    print('\nResources for computation tasks:', sum(alloc_tk)/n_episode_test)


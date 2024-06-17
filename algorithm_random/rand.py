import numpy as np
import os
import scipy.io
import my_env
import matplotlib.pyplot as plt

########################SETTING######################
lane_num = 3
n_veh = 5
width = 120

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

data_path = 'data/rand_data/n_%d'%n_veh

env = my_env.Environment(lane_num, n_veh, width)
env.make_new_game()

n_episode = 500
n_step_per_episode = 100

n_episode_test = 100  # test episodes

## Let's go
if IS_TRAIN:
    done = 0
    record_reward_average = []

    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)

        #env.make_new_game()
        if i_episode % 20 == 0:
            env.vehicle_renew_position() # update vehicle position
            env.renew_channel()
            env.R_V2I()


        state_old_all = []
        for i in range(n_veh):
            state = env.get_state()
            state_old_all.append(state)

        average_reward = 0
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, 2], dtype=np.float16)  # sub, power
            # receive observation
            action = np.random.randint(1,100/n_veh, 2*n_veh)
            action_all.append(action)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_veh):
                action_all_training[i, 0] = (action[0+i*2])/2 # chosen RB
                action_all_training[i, 1] = (action[1+i*2])/2
            action_channel = action_all_training.copy()
            train_reward = env.act_for_training(action_channel)

            record_reward[i_step] = train_reward

            # get new state
            for i in range(n_veh):
                state_new = env.get_state()
                state_new_all.append((state_new))

            # old observation = new_observation
            state_old_all = state_new_all
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
        print('step:', i_episode, 'reward', average_reward)

    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average
    np.save(data_path + '_rand_returns.pkl', record_reward_average)
    plt.figure(2)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


    print('Training Done. Saving models...')

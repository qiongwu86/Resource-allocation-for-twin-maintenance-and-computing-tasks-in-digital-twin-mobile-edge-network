import numpy as np
import matplotlib.pyplot as plt
import pickle

i = 5
maddpg_data = np.load('model/marl_model/marl_n_%d/returns.pkl.npy'%i)
sac_data = np.load('data/sac_data/n_%d/sac_returns.pkl.npy'%i)
#ppo_data = np.load('model/ppo_model/n_%dreturns.pkl.npy'%i)
#rand_data = np.load('data/rand_data/n_%d_rand_returns.pkl.npy'%i)
plt.figure(1)
plt.plot(maddpg_data, color = 'red',label = 'marl')
plt.plot(sac_data, color = 'blue',label = 'sac')
#plt.plot(ppo_data, color = 'orange',label = 'ppo')
#plt.plot(rand_data, color = 'green',label = 'rand')
plt.legend(loc='lower right')
plt.xlabel('episodes')
plt.ylabel('returns')
plt.show()
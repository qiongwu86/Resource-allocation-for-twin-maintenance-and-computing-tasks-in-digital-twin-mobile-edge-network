import numpy as np
import matplotlib.pyplot as plt
import pickle

i = 7
maddpg_data = np.load('model/marl_model/marl_n_%d/returns.pkl.npy'%i)
matd3_data = np.load('model/matd3_model/n_%d/MATD3_number_1_seed_0.npy'%i)
sac_data = np.load('data/sac_data/n_%d/sac_returns.pkl.npy'%i)
ppo_data = np.load('model/ppo_model/n_%d/agent_returns.pkl.npy'%i)
rand_data = np.load('data/rand_data/n_%d_rand_returns.pkl.npy'%i)
plt.figure(1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置英文为 Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12
plt.plot(maddpg_data, color = 'red',label = 'MADDPG')
plt.plot(matd3_data, color = 'c',label = 'MATD3')
plt.plot(sac_data, color = 'magenta',label = 'SAC')
plt.plot(ppo_data, color = 'blue',label = 'PPO')
plt.plot(rand_data, color = 'green',label = 'Rand')
plt.legend(loc='lower right')
plt.xlabel('episodes',fontsize=16)
plt.ylabel('returns',fontsize=16)
#plt.xlim(0,100)
a = np.mean(maddpg_data[200:])
b = np.mean(sac_data[200:])
c = np.mean(ppo_data[200:])
d = np.mean(rand_data[200:])
e = np.mean(matd3_data[200:])
print(a,b,c,d,e)
plt.show()
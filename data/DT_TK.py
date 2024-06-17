import numpy as np
import matplotlib.pyplot as plt

x = [5,6,7,8,9]
base = [0.5,0.5,0.5,0.5,0.5]
#marl_dt = [0.1768,0.2446,0.338,0.4238,0.524]
marl_tk = [0.082,0.1,0.156,0.1648,0.1798]
marl_tk_1 =[0.13,0.17,0.246,0.275,0.313] 
#sarl_dt = [0.3534,0.4128,0.4581,0.5464,0.6033]
#sarl_tk = [0.3454,0.5718,0.6324,0.6414,0.6843]
#ppo_dt = [0.2545, 0.3019, 0.3719, 0.4578, 0.9156]
#ppo_tk = [0.2123, 0.4921, 0.5681, 0.6538, 0.6721]
#rand_dt = [0.5992,0.7700,1.1795,1.4898,1.6039]
sarl_tk = [0.1529,0.1766,0.1898,0.2068,0.2535]
sarl_tk_1 = [0.1726,0.2143,0.263,0.2948,0.3346]
#plt.plot(x,marl_dt,'>',color = 'red', linestyle='-', label='marl-cstc_dt')
#plt.plot(x,sarl_dt,'v',color = 'blue', linestyle='-',label='sarl_dt')
#plt.plot(x,ppo_dt,'o',color = 'orange', linestyle='-',label='ppo_dt')
#plt.plot(x,rand_dt,'<',color = 'green', linestyle='-',label='rand_dt')
 
plt.plot(x,marl_tk,'>',color = 'red', linestyle='-', label='marl \u0394f=0.2')
plt.plot(x,sarl_tk,'v',color = 'blue', linestyle='-',label='sac \u0394f=0.2')
plt.plot(x,marl_tk_1,'>',color = 'red', linestyle='-.',label='marl \u0394f=-0.2')
plt.plot(x,sarl_tk_1,'v',color = 'blue', linestyle='-.',label='sac \u0394f=-0.2')
#plt.plot(x,ppo_tk,'o',color='orange',linestyle='-',label='ppo_tk')
#plt.plot(x,rand_tk,'<',color='green',linestyle='-',label='rand_tk')
#plt.plot(x,base,color = 'purple',label='time limit')
plt.ylim(0,0.5)
plt.xlabel('Number of vehicles')
#plt.ylabel('Time for Digital twins')
plt.ylabel('delay for Computation tasks')
plt.xticks(x)
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

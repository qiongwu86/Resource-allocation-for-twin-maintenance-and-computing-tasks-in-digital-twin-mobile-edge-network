import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['Times New Roman']

x = [4,5,6,7,8]
y = [0.2,0.4,0.6,0.8,1]
y2 = [0.98,0.99,1]
marl = [0.979, 0.9821, 1, 0.9952, 0.98]
matd3 = [0.4212, 0.4364, 0.4887, 0.5109, 0.6075]
sac = [0.3612, 0.4527, 0.54128, 0.6301, 0.7844]
ppo = [0.47395, 0.5521, 0.6114, 0.6981, 0.7637]
rand = [0.4804, 0.4745, 0.4804, 0.48673, 0.48091]
fig, ax = plt.subplots()
font2 = {'family':'Times New Roman','weight': 'normal', 'size': 12}
ax.plot(x,marl,'o',color = 'red', linestyle='-', label='TMTPRCS')
ax.plot(x,matd3,'p',color = 'c', linestyle='-', label='MATD3')
ax.plot(x,sac,'s',color = 'magenta', linestyle='--',label='SAC')
ax.plot(x,ppo,'D',color = 'blue', linestyle='-.',label='PPO')
ax.plot(x,rand,'^',color = 'green', linestyle=':',label='Rand')
plt.xticks(x)
plt.yticks(y, ['{:.0%}'.format(i) for i in y])
plt.xlabel('The number of vehicles', fontsize=16)
plt.ylabel('Resource utilization',fontsize=16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), fancybox=True, shadow=False, ncol=4, prop=font2)
plt.grid(True, linestyle='--')
plt.show()
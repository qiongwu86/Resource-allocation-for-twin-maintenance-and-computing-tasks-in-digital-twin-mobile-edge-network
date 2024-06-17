import numpy as np
import matplotlib.pyplot as plt
import pickle

x = [5,6,7,8,9]
#base = [0,0,0,0,0]
marl_game = [0.7208,0.6846,0.6240,0.6031,0.5555]
sac = [0.6123,0.4707,0.3936,0.3596,0.1395]
ppo = [0.6616,0.4648,0.3975,0.2966,0.1942]
rand = [0.537,0.455,0.3208,0.2297,0.1008]
plt.figure(1)
plt.xticks(x)
plt.plot(x,marl_game,'>', color = 'red', label = 'marl-cstc', linestyle = '-')
plt.plot(x,sac,'v', color = 'blue', label = 'sac', linestyle = '-')
plt.plot(x,ppo,'o', color = 'orange', label = 'ppo', linestyle = '-')
plt.plot(x,rand,'<', color = 'green', label = 'rand', linestyle = '-')
#plt.plot(x,base,linestyle='-')
plt.xlabel('Number of vehicles')
plt.ylabel('Utility')
plt.ylim(0,1)
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
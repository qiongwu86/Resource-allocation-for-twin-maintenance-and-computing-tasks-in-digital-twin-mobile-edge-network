import numpy as np
import matplotlib.pyplot as plt

x = [200,400,600,800,1000]
marl = [0.7208,0.7421,0.7457,0.7546,0.7750]
sarl = [0.6123,0.649,0.6494,0.655,0.657]
#ppo = [0.6506,0.653,0.6587,0.663,0.661]
#rand = [0.4988,0.518,0.5205,0.522,0.525]

plt.plot(x,marl,'>',color = 'red', linestyle='-', label='marl-cstc')
plt.plot(x,sarl,'v',color = 'blue', linestyle='-',label='sarl')
#plt.plot(x,ppo,'o',color = 'orange', linestyle='-',label='ppo')
#plt.plot(x,rand,'<',color = 'green', linestyle='-',label='rand')

plt.xlabel('transmission power(mW)')
plt.ylabel('Utility')
plt.xticks(x)
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim(0.6,0.8)
#plt.xlim(400,1000)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

marl_u =np.array([0.7208,0.6846,0.6240,0.6031,0.5555])
marl_r = np.array([0.995, 0.995, 0.99375, 0.9897, 0.9862])
u_r1 = np.divide(marl_u, marl_r)

sac_u = np.array([0.6123,0.4707,0.3936,0.3596,0.1395])
sac_r = np.array([0.8530,0.7943,0.8259,0.9170,0.8953])
u_r2 = np.divide(sac_u, sac_r)

ppo_u = np.array([0.6616,0.4648,0.3975,0.2966,0.1942])
ppo_r = np.array([0.9586,0.8717,0.8762,0.8974, 0.9068])
u_r3 = np.divide(ppo_u, ppo_r)

rand_u = np.array([0.537,0.455,0.3208,0.2297,0.1008])
rand_r = np.array([0.4738,0.4799,0.4908,0.4798,0.4946])
u_r4 = np.divide(rand_u, rand_r)

x = [5,6,7,8,9]
plt.plot(x,u_r1,'>', color = 'red', label = 'marl-cstc', linestyle = '-')
plt.plot(x,u_r2,'v', color = 'blue', label = 'sac', linestyle = '-')
plt.plot(x,u_r3,'o', color = 'orange', label = 'ppo', linestyle = '-')
#plt.plot(x,u_r4,'<', color = 'green', label = 'rand', linestyle = '-')
plt.xlabel('Number of vehicles')
plt.ylabel('The ratio of resources to utility')
plt.xticks(x)
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
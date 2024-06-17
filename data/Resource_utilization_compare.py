import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


x = [5,6,7,8,9]
y = [0.2,0.4,0.6,0.8,1]
y2 = [0.98,0.99,1]
base = [1,1,1,1,1]
marl = [0.995, 0.995, 0.99375, 0.9897, 0.9862]
sac = [0.8530,0.7943,0.8259,0.9170,0.8953]
ppo = [0.9586,0.8717,0.8762,0.8974, 0.9068]
rand = [0.4738,0.4799,0.4908,0.4798,0.4946]
plt.figure()
plt.subplot(1,2,1)
plt.plot(x,marl,'>',color = 'red', linestyle='-', label='marl-cstc')
plt.plot(x,sac,'v',color = 'blue', linestyle='-',label='sarl')
plt.plot(x,ppo,'o',color = 'orange', linestyle='-',label='ppo')
plt.plot(x,rand,'<',color = 'green', linestyle='-',label='rand')
plt.xticks(x)
plt.yticks(y, ['{:.0%}'.format(i) for i in y])
plt.xlabel('Number of vehicles')
plt.ylabel('resource utilization')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1,2,2)
plt.plot(x,marl,'>',color = 'red', linestyle='-')
plt.xticks(x)
plt.yticks(y2, ['{:.0%}'.format(i) for i in y2])
plt.xlabel('Number of vehicles')
#plt.plot(x,base)
#plt.ylim(0,0.8)

plt.grid(True, linestyle='--')
plt.show()
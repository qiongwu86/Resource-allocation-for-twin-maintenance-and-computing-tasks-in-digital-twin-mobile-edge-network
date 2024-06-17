import numpy as np
import matplotlib.pyplot as plt

#dt = np.array([55,54.9])#,52.6,23.6])
tk = np.array([78.904,30.2])#,42.8,23.7])
not_used = np.array([21.096,69.8])#,4.6,52.7])
x = range(len(tk))

algo = np.array(['marl', 'sac'])#, 'ppo', 'rand'])

#plt.bar(x, dt, alpha=0.7,color='red',label='resource for digital-twins',width=0.3)
plt.bar(x, tk, alpha=0.7,color='blue',label='resource for computation task',width=0.3)
plt.bar(x, not_used, alpha=0.7,color='green', label='unused resource',bottom=tk,width=0.3)
plt.ylabel('Resource Allocation of VEC server(GHZ/s)')
plt.legend(loc='center right', bbox_to_anchor=(1.1,1.05), fontsize=8)
plt.xticks(x,algo)
#plt.grid(True, linestyle='--')
plt.show()
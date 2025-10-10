import numpy as np
import matplotlib.pyplot as plt
import pickle

i = 7
maddpg_data = np.load('model/matd3_model/n_%d/MATD3_number_1_seed_0.npy'%i)
plt.figure(1)
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
plt.plot(maddpg_data, color = 'red',label = 'MATD3')
plt.legend(loc='lower right', prop=font2)
plt.xlabel('episodes')
plt.ylabel('returns')
#plt.xlim(0,100)
plt.show()
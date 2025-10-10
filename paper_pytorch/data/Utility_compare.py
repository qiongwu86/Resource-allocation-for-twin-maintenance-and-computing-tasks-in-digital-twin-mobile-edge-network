import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams['font.sans-serif'] = ['Times New Roman']
x = [4,5,6,7,8]
#base = [0,0,0,0,0]
marl = [0.9336, 0.8952, 0.8797, 0.8235, 0.7675]
matd3 = [0.792, 0.701, 0.643, 0.557, 0.472]
sac = [0.847, 0.8066, 0.7803, 0.7505, 0.654]
ppo = [0.844, 0.8053, 0.7689, 0.699, 0.645]
rand = [0.808, 0.722, 0.656, 0.578, 0.494]
fig, ax = plt.subplots()
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
plt.rcParams['xtick.labelsize'] = 12  # x 轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # y 轴刻度标签字体大小
plt.xticks(x)
ax.plot(x,marl,'o', color = 'red', label = 'TMTPRCS', linestyle = '-')
ax.plot(x,matd3,'p', color = 'c', label = 'MATD3', linestyle = '-')
ax.plot(x,sac,'s', color = 'magenta', label = 'SAC', linestyle = '--')
ax.plot(x,ppo,'D', color = 'blue', label = 'PPO', linestyle = '-.')
ax.plot(x,rand,'^', color = 'green', label = 'Rand', linestyle = ':')
#plt.plot(x,base,linestyle='-')
plt.xlabel('The number of vehicles',fontsize=16)
plt.ylabel('Utility',fontsize=16)
#plt.ylim(0,1)
plt.legend(loc='upper center', bbox_to_anchor=(0.6, 1.1), fancybox=True, shadow=False, ncol=4, prop=font2)
plt.grid(True, linestyle='--')

zoom_x_min = 4
zoom_x_max = 5
inset_ax = fig.add_axes([0.2, 0.15 , 0.4, 0.17])
inset_ax.plot(x, sac,'s', color = 'magenta', linestyle='--')
inset_ax.plot(x, ppo,'D', color = 'blue', linestyle='-.')

inset_ax.set_xlim(zoom_x_min, zoom_x_max)
inset_ax.set_ylim(0.804,0.848)
inset_ax.set_xticks([4,5])

rect = plt.Rectangle((zoom_x_min, 0.804), zoom_x_max-zoom_x_min, 0.0435,
                     edgecolor = (0,0,0,1), facecolor='none', linewidth=2)
ax.add_patch(rect)

from matplotlib.patches import ConnectionPatch
xy1 = (zoom_x_min, 0.8)
xy2 = (0, 1)
con1 = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=inset_ax, color="black")
ax.add_artist(con1)

xy1 = (zoom_x_max, 0.8)
xy2 = (1, 1)
con2 = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=inset_ax, color="black")
ax.add_artist(con2)


plt.show()
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
x = [200,400,600,800,1000]
marl = [0.7208,0.7421,0.7457,0.7546,0.7750]
sarl = [0.6123,0.649,0.6494,0.655,0.657]
ppo = [0.6506,0.653,0.6587,0.663,0.661]
rand = [0.4988,0.518,0.5205,0.522,0.525]

fig, ax = plt.subplots()
ax.plot(x,marl,'o',color = 'red', linestyle='-', label='marl-cstc')
ax.plot(x,sarl,'s',color = 'blue', linestyle='-.',label='sac')
ax.plot(x,ppo,'D',color = 'orange', linestyle='--',label='ppo')
ax.plot(x,rand,'^',color = 'green', linestyle=':',label='rand')

plt.xlabel('传输功率')
plt.ylabel('效能')
plt.xticks(x)
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim(0.4,0.8)

zoom_x_min = 400
zoom_x_max = 1000
inset_ax = fig.add_axes([0.4, 0.4 , 0.4, 0.15])
inset_ax.plot(x, sarl,'s', color = 'blue', linestyle='-.', label = 'sac')
inset_ax.plot(x, ppo,'D', color = 'orange', linestyle='--', label = 'ppo')

inset_ax.set_xlim(zoom_x_min, zoom_x_max)
inset_ax.set_ylim(0.648,
                  0.665)
inset_ax.set_xticks([400, 600, 800, 1000])

rect = plt.Rectangle((zoom_x_min, 0.648), zoom_x_max-zoom_x_min, 0.02,
                     edgecolor = 'black', facecolor='none')
ax.add_patch(rect)

from matplotlib.patches import ConnectionPatch
xy1 = (zoom_x_min, 0.648)
xy2 = (0, 1)
con1 = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=inset_ax, color="black")
ax.add_artist(con1)

xy1 = (zoom_x_max, 0.648)
xy2 = (1, 1)
con2 = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=inset_ax, color="black")
ax.add_artist(con2)

plt.show()
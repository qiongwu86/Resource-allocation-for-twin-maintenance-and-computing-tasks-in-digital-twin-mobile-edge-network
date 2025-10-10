import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams['font.sans-serif'] = ['Times New Roman']

x = [4,5,6,7,8]
#base = [100,100,100,100,100]
marl_tk = [26.70, 29.27, 36.51, 41.72, 47.84]
matd3_tk= [90.32, 112.27, 127.35, 146.33, 158.92]
sarl_tk = [59.21, 62.50, 63.49, 63.78, 64.13]
ppo_tk = [55.95, 60.87, 62.07, 62.43, 62.93]
rand_tk = [86.72, 109.54, 125.90, 143.23, 157.10]
#--------------------------------------------------
fig, ax = plt.subplots()
#font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 9}

# 绘制主图
ax.plot(x, marl_tk, color='red', label='TMTPRCS', linestyle='-', marker='o', linewidth=1.5)
ax.plot(x, matd3_tk, color='c', label='MATD3', linestyle='-', marker='p', linewidth=1.5)
ax.plot(x, sarl_tk, color='magenta', label='SAC', linestyle='--', marker='s', linewidth=1.5)
ax.plot(x, ppo_tk, color='blue', label='PPO', linestyle='-.', marker='D', linewidth=1.5)
ax.plot(x, rand_tk, color='green', label='Rand', linestyle=':', marker='^', linewidth=1.5)
#ax.plot(x, base, linestyle='-', color='black', linewidth=3)

# 标题和标签
#ax.set_title('DT Maintenance Delay')
plt.xticks(x)
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
ax.set_xlabel('The number of vehicles',fontsize=16)
ax.set_ylabel('Delay of tasks processing (ms)',fontsize=16)
ax.legend(loc='upper left', bbox_to_anchor=(0.01, 1.02),prop=font2)
plt.grid(True, linestyle='--')

zoom_x_min = 5
zoom_x_max = 8
inset_ax = fig.add_axes([0.4, 0.41 , 0.4, 0.15])
inset_ax.plot(x, sarl_tk,'s', color = 'magenta', linestyle='--')
inset_ax.plot(x, ppo_tk,'D', color = 'blue', linestyle='-.')

inset_ax.set_xlim(zoom_x_min, zoom_x_max)
inset_ax.set_ylim(60,65)
inset_ax.set_xticks([5,6,7,8])

rect = plt.Rectangle((zoom_x_min, 60), zoom_x_max-zoom_x_min, 6,
                     edgecolor = (0,0,0,1), facecolor='none', linewidth=2)
ax.add_patch(rect)

from matplotlib.patches import ConnectionPatch
xy1 = (zoom_x_min, 60)
xy2 = (0, 1)
con1 = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=inset_ax, color="black")
ax.add_artist(con1)

xy1 = (zoom_x_max, 60)
xy2 = (1, 1)
con2 = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="axes fraction",
                       axesA=ax, axesB=inset_ax, color="black")
ax.add_artist(con2)

# 显示图形
plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams['font.sans-serif'] = ['Times New Roman']

x = [4,5,6,7,8]
base = [100,100,100,100,100]
marl_dt = [8.51, 17.39, 18.94, 33.75, 53.47]
matd3_dt = [23.05, 34.33, 39.42, 48.63, 59.43]
sarl_dt = [21.49, 31.41, 37.66, 46.70, 55.64]
ppo_dt = [19.49, 28.90, 37.34, 45.79, 56.25]
rand_dt = [31.05, 54.67, 75.38, 103.6, 140.4]
#--------------------------------------------------
fig, ax = plt.subplots()
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

# 绘制主图
ax.plot(x, marl_dt, color='red', label='TMTPRCS', linestyle='-', marker='o', linewidth=1.5)
ax.plot(x, matd3_dt, color='c', label='MATD3', linestyle='-', marker='p', linewidth=1.5)
ax.plot(x, sarl_dt, color='magenta', label='SAC', linestyle='--', marker='s', linewidth=1.5)
ax.plot(x, ppo_dt, color='blue', label='PPO', linestyle='-.', marker='D', linewidth=1.5)
ax.plot(x, rand_dt, color='green', label='Rand', linestyle=':', marker='^', linewidth=1.5)
ax.plot(x, base, linestyle='-', color='black',label='Deadline', linewidth=3)

# 标题和标签
#ax.set_title('DT Maintenance Delay')
plt.xticks(x)
ax.set_xlabel('The number of vehicles',fontsize=16)
ax.set_ylabel('Delay of twin maintenance (ms)',fontsize=16)
ax.legend(loc='upper left', bbox_to_anchor=(0.01, 1.12), prop=font2)
plt.grid(True, linestyle='--')

# 显示图形
plt.show()
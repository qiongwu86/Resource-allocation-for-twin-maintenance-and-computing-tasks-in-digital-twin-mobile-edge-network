import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 数据
categories = [4, 5, 6, 7, 8]
#marl
marl_tk = [47.65, 52.5, 51, 52.37, 52]

#matd3
matd3_tk = [14.12, 22.32, 23.43, 23.56, 24.32]

#sac
sac_tk = [18.05, 22.65, 27.062, 31.50, 36.09]

#ppo
ppo_tk = [22.394, 23.24, 33.10, 35.33, 38.36]

#rand
rand_tk = [23.95, 23.70, 24.03, 24.353, 24.061]

# X轴位置
x = np.arange(len(categories))

# 柱状图的宽度
width = 0.15

# 绘制堆叠柱状图
fig, ax = plt.subplots()

alpha_adjust = 0.4

font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

# 绘制每组堆叠柱状图
ax.bar(x, marl_tk, width, color='#ff9999', label='TMTPRCS', zorder=3)

ax.bar(x-width, sac_tk, width, color='#ff66ff', label='SAC', zorder=3)

ax.bar(x-width*2, ppo_tk, width, color='lightblue', label='PPO', zorder=3)

ax.bar(x-width*3, rand_tk, width, color='lightgreen', label='Rand',zorder=3)

ax.bar(x-width*4, matd3_tk, width, color='c', label='MATD3', zorder=3)

# 添加标签和标题
ax.set_xlabel('The number of vehicles',fontsize=16)
ax.set_ylabel('Resources for tasks processing (Ghz)',fontsize=16)
#ax.set_title('Stac')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), fancybox=True, shadow=False, ncol=4, prop=font2)
plt.grid(which="both", linestyle='--', zorder=0)

# 显示图表
plt.show()
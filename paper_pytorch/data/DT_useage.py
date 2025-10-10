import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 数据
categories = [4, 5, 6, 7, 8]
#marl
marl_dt = [50.25, 45.71, 49, 47.16, 46]

#matd3
matd3_dt= [25, 21.32, 25.44, 27.53, 36.43]

#sac
sac_dt = [18.07, 22.62, 27.066, 31.51, 42.35]

#ppo
ppo_dt = [25.001, 31.97, 28.04, 34.48, 38.01]

#rand
rand_dt = [24.09, 23.75, 24.01, 24.32, 24.03]

# X轴位置
x = np.arange(len(categories))

# 柱状图的宽度
width = 0.15

# 绘制堆叠柱状图
fig, ax = plt.subplots()

alpha_adjust = 0.4

font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

# 绘制每组堆叠柱状图
ax.bar(x, marl_dt, width, color='#ff9999', label='TMTPRCS', zorder=3)

ax.bar(x-width, sac_dt, width, color='#ff66ff', label='SAC', zorder=3)

ax.bar(x-width*2, ppo_dt, width, color='lightblue', label='PPO', zorder=3)

ax.bar(x-width*3, rand_dt, width, color='lightgreen', label='Rand',zorder=3)

ax.bar(x-width*4, matd3_dt, width, color='c', label='MATD3',zorder=3)

# 添加标签和标题
ax.set_xlabel('The number of vehicles',fontsize=16)
ax.set_ylabel('Resources for twin maintenance (Ghz)',fontsize=16)
#ax.set_title('Stac')
ax.set_xticks(x)
ax.set_xticklabels(categories)
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=False, prop=font2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), fancybox=True, shadow=False, ncol=4, prop=font2)
plt.grid(which="both", linestyle='--', zorder=0)

# 显示图表
plt.show()

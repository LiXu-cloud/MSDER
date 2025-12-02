import matplotlib.pyplot as plt
import pandas as pd

# 从 CSV 文件读取数据
df = pd.read_csv('2025-09-12_23-15-19_avg_rewards.csv')
# 绘制图形
plt.plot(df['Episode'], df['Avg Reward'], marker='o')
plt.title('Episode vs Average Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid()
plt.savefig('avg_reward_plot.png')  # 保存图形
plt.show()  # 显示图形
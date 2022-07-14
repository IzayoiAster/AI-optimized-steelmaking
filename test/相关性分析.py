# 相关性分析
import matplotlib.pyplot as plt
import numpy as np

source_data = np.loadtxt('../成锭率.csv', encoding='utf-8-sig', delimiter=',')
print("原始数据集：", source_data.shape)
print("-------------------------------------------------------")

fig, subs = plt.subplots(4, 4)

for i in range(14):
    subs[i // 4][i % 4].scatter(source_data[:, i], source_data[:, 14])

plt.show()

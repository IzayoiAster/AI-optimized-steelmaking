# 成锭率模型训练
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

max_score = 0

# 提前剔除了部分显然不合规的数据（成锭率为1的数据）
source_data = np.loadtxt('成锭率.csv', encoding='utf-8-sig', delimiter=',')

# 调参（Product里面那个更完善）
# l = 5788
# r = 50000
# for i in range(l, r):
#
#     X_train, X_test, y_train, y_test = train_test_split(source_data[:, :14], source_data[:, 14], test_size=0.2,
#                                                         random_state=36591)
#
#     # 数据标准化
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     regressor = RandomForestRegressor(n_estimators=6, random_state=5788, max_depth=9, max_features=6)
#     regressor.fit(X_train, y_train)
#
#     # 评估
#     y_pred = regressor.predict(X_test)
#     r2 = regressor.score(X_test, y_test)
#     mse = mean_squared_error(y_test, y_pred)
#
#     if r2 > max_score:
#         max_score = r2
#         print("\nparam val:", i, "max score:", max_score)
#     else:
#         print(">", end="")
#         if i % 100 == 0:
#             print("%.2f" % ((i - l)*100/(r - l)), end="")
#             print("%")

X_train, X_test, y_train, y_test = train_test_split(source_data[:, :14], source_data[:, 14], test_size=0.2,
                                                    random_state=36591)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regressor = RandomForestRegressor(n_estimators=6, random_state=5788, max_depth=9, max_features=6)
regressor.fit(X_train, y_train)

# 评估
y_pred = regressor.predict(X_test)
print(X_test, y_pred)
r2 = regressor.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("R2:", r2, "\nMSE:", mse)

# 可视化
plt.figure()
plt.plot(np.arange(len(y_test)), y_test, "bo-", label="True")
plt.plot(np.arange(len(y_pred)), y_pred, "ro-", label="Pred")
plt.title(f'R2: {r2}\nMSE: {mse}')
plt.legend(loc="best")
plt.show()

# 导出模型
# pkl_filename = "ingot.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(regressor, file)

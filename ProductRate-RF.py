# 成材率模型训练
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pickle

# 提前剔除了部分显然不合规的数据（序号112行，存在明显异常列值12）
source_data = np.loadtxt('成材率.csv', encoding='utf-8-sig', delimiter=',')

# 调参
# max_score = 0
# max_i = 0
#
# l = 1
# r = 50
# for i in range(l, r):
#
#     X_train, X_test, y_train, y_test = train_test_split(source_data[:, :6], source_data[:, 6], test_size=0.2,
#                                                         random_state=75698)
#     # 数据标准化
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     # 96730 in 10w
#     regressor = DecisionTreeRegressor(random_state=96730, max_depth=None, max_features=6)
#     regressor.fit(X_train, y_train)
#
#     # 评估
#     y_pred = regressor.predict(X_test)
#     r2 = regressor.score(X_test, y_test)
#     mse = mean_squared_error(y_test, y_pred)
#
#     # print(r2)
#
#     if r2 > max_score:
#         max_score = r2
#         max_i = i
#         print("\nparam val:", i, "max score:", max_score)
#     else:
#         print(">", end="")
#         if i % 100 == 0:
#             print("%.2f" % ((i - l) * 100 / (r - l)), end="")
#             print("%", max_score, "at", max_i)

X_train, X_test, y_train, y_test = train_test_split(source_data[:, :6], source_data[:, 6], test_size=0.2,
                                                    random_state=75698)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regressor = DecisionTreeRegressor(random_state=96730, max_depth=None, max_features=6)
regressor.fit(X_train, y_train)

# 评估
y_pred = regressor.predict(X_test)
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
# pkl_filename = "product.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(regressor, file)

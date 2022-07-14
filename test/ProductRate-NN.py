# 成锭率
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# https://blog.csdn.net/idwtwt/article/details/100328331
source_data = np.loadtxt('../成材率.csv', encoding='utf-8-sig', delimiter=',')

# print("原始数据集大小：", source_data.shape)
# print("----------------------------------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(source_data[:, :6], source_data[:, 6], test_size=0.2,
                                                    random_state=0)

# 数据标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# print("X训练集：\n", X_train)
# print("y训练集：\n", y_train)
# print("----------------------------------------------------------------------")
# print("X测试集：\n", X_test)
# print("y测试集：\n", y_test)
# print("----------------------------------------------------------------------")


class LinearModel(torch.nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.nn1 = torch.nn.Linear(6, 4)
        self.nn2 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.nn1(x))
        x = self.sigmoid(self.nn2(x))
        return x


model = LinearModel()

# 优化神经网络
# https://blog.csdn.net/qq_34690929/article/details/79932416
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 均方根误差
# https://blog.csdn.net/snake_seeker/article/details/108685969
# https://blog.csdn.net/qq_23123181/article/details/124092355
# 与L2 MSE相比，L1对于异常值有更好的鲁棒性
loss = torch.nn.L1Loss()
train_loss_all = []

# https://blog.csdn.net/weixin_39504171/article/details/103179067
for epoch in range(1000):
    # flatten：扁平化（变为1维）
    y_train_pred = model(X_train).flatten()
    train_loss = loss(y_train_pred, y_train)
    train_loss_all.append(train_loss.detach().numpy())
    # 将模型的参数梯度初始化为0
    optimizer.zero_grad()
    # 反向传播计算梯度
    train_loss.backward()
    # 更新所有参数
    optimizer.step()

y_test_pred = model(X_test).flatten()
print("预测值：", y_test_pred.detach().numpy())
print("实际值：", y_test.detach().numpy())
r2 = r2_score(y_test.detach().numpy(), y_test_pred.detach().numpy())
print("R2: ", r2)

if r2 >= 0.52:
    torch.save(model, './IngotRate.pkl')
    print("Model saved.")

# plt.plot(train_loss_all)
# plt.title("MSE Loss")
# # 迭代次数
# plt.xlabel("Epoch")
# # 损失值
# plt.ylabel("Loss")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

boston = np.loadtxt('housing.data')
# print("原始数据集：", boston.shape)
# print("-------------------------------------------------------")

# 计算相关系数矩阵
# corr = np.corrcoef(boston, rowvar=False)
# print(corr[13])

X_train, X_test, y_train, y_test = train_test_split(boston[:, :13], boston[:, 13], test_size=0.2, random_state=3)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# print("X训练集：\n", X_train)
# print("y训练集：\n", y_train)
# print("-------------------------------------------------------")


class LinearModel(torch.nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        # 第一层
        self.h1 = torch.nn.Linear(in_features=13, out_features=8)
        # 第二层
        self.h2 = torch.nn.Linear(in_features=8, out_features=5)
        # 第三层
        self.line = torch.nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        # 第一层
        x = self.h1(x)
        x = F.relu(x)
        # 第二层
        x = self.h2(x)
        x = F.relu(x)
        # 第三层
        out = self.line(x)
        return out


model = LinearModel()

# 优化神经网络
# https://blog.csdn.net/qq_34690929/article/details/79932416
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 0.01

# 均方根误差
# https://blog.csdn.net/snake_seeker/article/details/108685969
loss = torch.nn.MSELoss()
train_loss_all = []

# https://blog.csdn.net/weixin_39504171/article/details/103179067
for i in range(100):
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
r2 = r2_score(y_test.detach().numpy(), y_test_pred.detach().numpy())
print("R2: ", r2)

if r2 >= 0.6:
    torch.save(model, './Model.pkl')
    print("Model saved.")

plt.plot(train_loss_all)
plt.show()

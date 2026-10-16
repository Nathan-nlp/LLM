import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据：X 是输入 (x1, x2)，y 是目标输出 (2*x1 + 3*x2)
X = torch.randn(100, 2)
y = 2 * X[:, 0:1] + 3 * X[:, 1:2]

# 2. 定义神经网络架构
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Linear(2, 4)   # 隐藏层：输入维度 2，输出维度 4
        self.relu = nn.ReLU()           # 激活函数：引入非线性
        self.output = nn.Linear(4, 1)   # 输出层：输入维度 4，输出维度 1

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# 3. 实例化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()                  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01) # 随机梯度下降优化器

# 4. 训练循环 (Training Loop)
for epoch in range(100):
    optimizer.zero_grad()                 # 第一步：清空之前的梯度
    predictions = model(X)                # 第二步：前向传播，计算预测值
    loss = criterion(predictions, y)      # 第三步：计算损失 (Loss)
    loss.backward()                       # 第四步：反向传播，计算梯度
    optimizer.step()                      # 第五步：更新模型参数

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

# 5. 验证预测效果
test_input = torch.tensor([[1.0, 2.0]])   # 真实结果应该是 2*1 + 3*2 = 8.0
predicted = model(test_input)
print(f"\n测试输入 [1.0, 2.0] -> 神经网络预测值: {predicted.item():.4f}")

import torch
import torch.nn as nn
import torchvision                          #torch中用来处理图像的库
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
#设置一些超参
num_epochs = 2        #训练的周期
batch_size = 100      #批训练的数量
learning_rate = 0.001 #学习率（0.1,0.01,0.001）
#
# 导入训练数据
train_dataset = datasets.MNIST(root='G:/dataset/',  # 数据集保存路径
                               train=True,  # 是否作为训练集
                               transform=transforms.ToTensor(),  # 数据如何处理, 可以自己自定义
                               download= False)  # 路径下没有的话, 可以下载

# 导入测试数据
test_dataset = datasets.MNIST(root='G:/dataset/',
                              train=False,
                              transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, #分批
                                           batch_size=batch_size,
                                           shuffle=True)          #随机分批

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class MLP(nn.Module):                    #继承nn.module
    def __init__(self):
        super(MLP, self).__init__()      #继承的作用
        self.layer1 = nn.Linear(784,300)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(300,10)

    def forward(self,x):             #网络传播的结构
        x = x.reshape(-1, 28*28)
        x = self.layer1(x)
        x = self.relu(x)
        y = self.layer2(x)
        return y

mlp = MLP() #类的实例化
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = mlp(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向求梯度
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


#测试模型
mlp.eval()      #测试模式，关闭正则化
correct = 0
total = 0
for images, labels in test_loader:
    outputs = mlp(images)
    _, predicted = torch.max(outputs, 1)   #返回值和索引
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('测试准确率: {:.4f}'.format(100.0*correct/total))






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch import Tensor, optim
    import torch.nn.functional as F
    import torch.nn as nn
    from time import time
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    from typing import Type, Any, Callable, Union, List, Optional

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 让数据转化为张量
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 进行归一化

    batch_size = 16  # 一次抓取几个样本进行训练

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 第一层网络 卷积核3，输出32层
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 第二层网络
        self.pool2 = nn.MaxPool2d(2, 2)  # 池化层
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 第三层网络
        self.pool3 = nn.MaxPool2d(2, 2)  # 池化层

        self.fc1 = nn.Linear(128 * 4 * 4, 625)
        self.fc2 = nn.Linear(625, 10)  # 第二个线性函数是输出十个得分

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # 在卷积后进行激活函数的运算
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        #  print(x.shape)
        x = x.view(-1, 128 * 4 * 4 )
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net().to(device)  # 放到gpu运行，已经运行构造函数了

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001 这里优化是用随机梯度下降和动量，可以改成adam算法

num_epochs = 1000  # 执行1000次

for epoch in range(num_epochs):
    sum_loss = 0.0  # 一开始loss为0
    for i, data in enumerate(train_loader):  # 这里就开始载入训练数据集了
        images, labels = data
        images = images.to(device)  # 到gpu上面运行
        labels = labels.to(device)  # 同上

        # Forward pass
        outputs = F.softmax(net(images))  # 输出归一化
        loss = criterion(outputs, labels)  # 用交叉熵损失函数来对输出和标签进行判断

        # Backward and optimize
        optimizer.zero_grad()  # 反向传播和优化
        loss.backward()  # 自动计算梯度
        optimizer.step()  # 调整位置

        # calculate loss
        sum_loss += loss.item()  # loss赋值到sum_loss中
        if i % 100 == 99:
            print('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0
net.eval()  # 表示将各个参数都固定住
correct = 0
total = 0
for data_test in test_loader:
    images, labels = data_test  # 解包
    images = images.to(device)
    labels = labels.to(device)
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()
print("correct1: ", correct)
print("Test acc: {0}".format(correct.item() / len(test_set)))
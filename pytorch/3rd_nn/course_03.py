
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

'''
一个典型的神经网络训练过程包括以下几点：
    1.定义一个包含可训练参数的神经网络
    2.迭代整个输入
    3.通过神经网络处理输入
    4.计算损失(loss)
    5.反向传播梯度到神经网络的参数
    6.更新网络的参数，典型的用一个简单的更新方法：weight = weight - learning_rate *gradient
'''

'''
 `nn.Conv2d`、`nn.Linear`、`F.max_pool2d`、`F.relu` 这四个函数/模块进行对比说明：

| 名称            | 类型         | 主要作用             | 输入形状                  | 主要参数/说明                   | 常见用途                  |
|-----------------|--------------|----------------------|---------------------------|----------------------------------|---------------------------|
| `nn.Conv2d`     | 模块（层）   | 二维卷积             | (B, C, H, W)              | 输入/输出通道数、卷积核大小等    | 图像特征提取              |
| `nn.Linear`     | 模块（层）   | 全连接（线性变换）   | (B, features)             | 输入/输出特征数                 | 分类、特征变换            |
| `F.max_pool2d`  | 函数         | 最大池化             | (B, C, H, W)              | 池化窗口大小、步长等            | 降低空间尺寸，提取主特征   |
| `F.relu`        | 函数         | 激活函数（ReLU）     | 任意形状                  | 无                              | 非线性变换，增加表达能力   |

---

### 详细说明

- **nn.Conv2d**：卷积层，自动带参数（权重和偏置），用于提取空间局部特征。
- **nn.Linear**：全连接层，自动带参数（权重和偏置），用于特征的线性组合。
- **F.max_pool2d**：最大池化操作，不带参数，仅做下采样，保留每个池化窗口的最大值，常用于降低特征图尺寸。
- **F.relu**：激活函数，不带参数，将输入中的负数变为0，增加网络的非线性表达能力。

---

**简单理解：**

- `nn.Conv2d`、`nn.Linear` 是“带参数”的网络层，常用于特征提取和变换。
- `F.max_pool2d`、`F.relu` 是“无参数”的操作，常用于特征处理和激活。
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


'''
    
'''

if __name__ == '__main__':
    net = Net()
    logger.info(net)

    '''
        0.模型的可训练参数：
    '''
    params = list(net.parameters())
    # logger.info(f"params: {params}")
    logger.info(len(params))
    
    # logger.info(params[0])  # conv1's .weight
    # logger.info(params[0].size())  # conv1's .weight

    '''
        1.前向传播
            神经网络的输入数据：B, C, H, W
    '''
    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    logger.info(output.shape) # torch.Size([1, 10])

    '''
        2.损失函数:
            MSELoss： 均方误差
    '''
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    logger.info(loss)
    
    '''
        计算图：
            input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
                -> view -> linear -> relu -> linear -> relu -> linear
                -> MSELoss
                -> loss
    '''
    logger.info(loss.grad_fn)  # MSELoss
    logger.info(loss.grad_fn.next_functions[0][0])  # Linear
    logger.info(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    '''
        3. 反向传播:
            反向传播后自动计算梯度保存在每一层参数的 net.conv1.bias.grad 属性中;
            这个梯度和 net.parameters() 中的f.grad.data是一样的。
    '''
    net.zero_grad()     # zeroes the gradient buffers of all parameters

    logger.info('conv1.bias.grad before backward')

    for idx, f in enumerate(net.parameters()):
        if f is net.conv1.bias:
            # logger.info(f.grad.data)
            logger.info(net.conv1.bias.grad)

    loss.backward()

    logger.info('conv1.bias.grad after backward')

    for idx, f in enumerate(net.parameters()):
        if f is net.conv1.bias:
            # logger.info(f)
            logger.info(f.grad.data)
            logger.info(net.conv1.bias.grad)

    '''
        4. 更新权重:
            weight = weight - learning_rate * gradient
    '''
    learning_rate = 0.01
    for idx, f in enumerate(net.parameters()):
        f.data.sub_(f.grad.data * learning_rate)
    
    '''
        5. 当使用不同的更新规则，类似于 SGD, Nesterov-SGD, Adam, RMSProp, 等
    '''
    import torch.optim as optim

    optimizer = optim.SGD(net.parameters(), lr=0.01)    # 创建权重更新算法
    optimizer.zero_grad()   # 梯度清空
    output = net(input)     # 前向传播
    loss = criterion(output, target) # 计算损失
    loss.backward()         # 反向传播
    optimizer.step()        # 更新权重

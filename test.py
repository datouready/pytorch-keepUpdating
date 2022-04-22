import torch
import torch.nn.functional as F
 
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.ReLU(x)
        x = F.max_pool2d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)
print(9)
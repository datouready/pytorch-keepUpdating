参考：
1、https://blog.csdn.net/HiWangWenBing/article/details/120614234
[Pytorch系列-30]：神经网络基础 - torch.nn库五大基本功能：
nn.Parameter、nn.Linear、nn.functioinal、nn.Module、nn.Sequentia

第一章、 torch.nn 简介
1.1 torch.nn相关库的导入
#环境准备
import numpy as np  #numpy库
import math         #数学运算库
import matplotlib.pyplot as plt #画图库

import torch #torch基础库
import torch.nn  as nn #torch神经网络库
import torch.nn.functional as F

1.2 torch.nn概述
Pytoch提供了几个设计得非常棒的模块和类，比如torch.nn,torch.optim,Dataset,DataLoader来帮助程序员设计和训练神经网络
nn是neural Network的简称，帮助程序员方便执行如下与神经网络相关的行为：
（1）创建神经网络
（2）训练神经网络
（3）保存神经网络
（4）恢复神经网络

其包括如下五大基础功能模块：
torch.nn库
torch.nn是专门为神经网络设计的模块化接口。
nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Parameter
nn.Linear
nn.functional
nn.Module
nn.Sequential

第二章 nn.Linear类（全连接层）
2.1 函数功能
用于创建一个多输入、多输出的全连接层
nn.Linear本身并不包含激活函数（Functional）

Class torch.nn.Linear(in_features,out_features,bias=True)

in_features:
指的是输入的二维张量的大小，即输入的[batch_size,input_size]中的size
in_features的数量，决定的参数的个数  Y=WX+b,X的维度就是in_features,X的维度决定的W的维度，总的参数个数=in_features+1

out_features:
指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size,output_size]
out_features的数量，决定了全连接层中神经元的个数，因为每个神经元只有一个输出
多个输出，就需要多个神经元

2.4 使用nn.Linear类创建全连接层
# nn.Linear
# 建立单层的多输入，多输出全连接层
# in_features由输入张量的形状决定，out_features则决定了输出张量的形状
full_connect_layer=nn.Linear(in_features=28*28*1,out_features=3)
print("full_connect_layer:",full_connect_layer)
print("parameters:",full_connect_layer.parameters)

# 假定输入的图像形状为[28*28*1]
x_input=torch.randn(1,28,28,1)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入
x_input=x_input.view(1,28*28*1)
print("x_input.shape",x_input.shape)

# 调用全连接层
y_output=full_connect_layer(x_input)
print("y_output.shape",y_output.shape)
print("y_output:",y_output)

输出：
full_connect_layer: Linear(in_features=784, out_features=3, bias=True)
parameters        : <bound method Module.parameters of Linear(in_features=784, out_features=3, bias=True)>
x_input.shape: torch.Size([1, 784])
y_output.shape: torch.Size([1, 3])
y_output: tensor([[-0.2892, -0.3084,  0.9027]], grad_fn=<AddmmBackward>)

第三章 nn.functional(常见函数)

## 3.1 nn.functional概述

nn.functional定义了创建神经网络所需要的一些常见的处理函数。如没有激活函数的神经元，各种激活函数等。
nn.functional
包含torch.nn库中所有函数，包含大量loss和activate function
torch.nn.functional.conv2d(input,weight,bias=None,stride=1,padding=0,dilation=1,groups=1)
nn.functional.xxx是函数接口
nn.functional.xxx无法与nn.Sequential结合使用
没有学习参数的（eg,maxpool,loss_func,activate func）等根据个人选择使用nn.functional.xxx或nn.Xxx
需要特别注意dropout层

# 3.2 nn.functional函数分类
nn.functional包括神经网络前向和后向处理所需要的常见函数。
（1）神经元处理函数
（2）激活函数

3.3 激活函数的案例
（1） relu案例

## nn.functional.relu()
print(y_output)
out=nn.functional.relu(y_output)
print(out.shape)
print(out)

tensor([[ 0.1023,  0.7831, -0.2368]], grad_fn=<AddmmBackward>)
torch.Size([1, 3])
tensor([[0.1023, 0.7831, 0.0000]], grad_fn=<ReluBackward0>)

(2)sigmoid案例
# nn.functional.sigmoid()
print(y_output)
out=nn.functional.sigmoid(y_output)
print(out.shape)
print(out)

第四章 nn.xxx和nn.functional.xxx比较

4.1 相同点
nn.Xxx和nn.functional.xxx的实际功能是相同的，即nn.Conv2d和nn.functional.conv2d都是进行卷积，nn.Dropout和nn.functional.dropout都是进行dropout
运行效率也是近乎相同

4.2不同点
nn与nn.functional有什么区别？
nn.functional是函数接口
nn.Xxx是.nn.functional.xxx的类封装，并且nn.Xxx都继承于一个共同祖先nn.Module
nn.Xxx除了具有nn.functional.xxx功能之外，内部附带nn.Module相关的属性和方法，比如eg.train(),eval(),load_state_dict,state_dict
nn.Xxx继承于nn.Module,能够很好的与nn.Sequential结合使用，而nn.functional.xxx无法与nn.Sequential.xxx结合使用
nn.Xxx需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。nn.functional.xxx同时传入输入的数据和weight,bias等其他参数。
nn.Xxx不需要自己定义和管理weight;而nn.functional.xxx需要你自己定义weight，每次调用的时候都需要手动传入weight,不利于代码复用

第五章 nn.Parameter类
5.1 nn.Parameter概述
Parameter实际上也是Tensor,也就是一个多维矩阵，是Variable类中的一个特殊类
当我们创建一个mode时，nn会自动创建相应的参数parameter,并会自动累加到模型的Parameter成员列表中
Parameters VS buffers
一种时反向传播需要被optimizer更新的，称之为parameter
self.register_parameter("param",param)
self.param=nn.Parameter(torch.randn(1))
一种是反向传播不需要被optimizer更新，称之为buffer
self.register_buffer('my_buffer',torch.randn(1))

# 5.2单个全连接层中参数的个数
in_features的数量，决定参数的个数  Y=WX+b,X的维度就是in_feature,X的维度决定W的维度，总的参数个数=in_features+1
out_features的数量，决定了全连接层中神经元的个数，因为每个神经元只有一个输出
多少个输出，就需要多少个神经元
总的W的参数的个数=in_features*out_features
总的b参数的个数=1*out_features
总的参数（W和B）的个数=（in_features+1）*out_features

# 5.3 使用参数创建全连接层代码案例
### nn.functional.linear()
x_input=torch.Tensor([1.,1.,1.])
print("x.input.shape:",x_input.shape)
print("x.input:",x_input)
print("")

Weights1=nn.Parameter(torch.rand(3))
print("Weights.shape",Weights.shape)
print("Weight:",Weights1)
print("")

Bias1=nn.Parameter(torch.rand(1))
print("Bias.shape:",Bias1.shape)
print("Bias:",Bias1)
print("")

Weights2=nn.Parameter(torch.Tensor(3))
print("Weights.shape:", Weights2.shape)
print("Weights      :", Weights2)



print("\nfull_connect_layer")
full_connect_layer=nn.functional.linear(x_input,Weights1)
print(full_connect_layer)

输出：
x_input.shape: torch.Size([3])
x_input      : tensor([1., 1., 1.])

Weights.shape: torch.Size([3])
Weights      : Parameter containing:
tensor([0.3339, 0.7027, 0.9703], requires_grad=True)

Bias.shape: torch.Size([1])
Bias      : Parameter containing:
tensor([0.4936], requires_grad=True)

Weights.shape: torch.Size([3])
Weights      : Parameter containing:
tensor([0.0000e+00, 1.8980e+01, 1.1210e-44], requires_grad=True)

full_connect_layer
tensor(2.0068, grad_fn=<DotBackward>)

第六章 nn.Module类
nn.Module
它是一个抽象概念，既可以表示神经网络中的某个层（layer）,也可以表示一个包含很多层的神经网络
model.parameters()
model.buffers()
model.state_dict()
model.modules()
forward(),to()

第七章 利用nn.Sequential类创建神经网络（继承与nn.Module类）

7.1概述
A sequential container. Modules will be added to it in the order they are passed in the constructor. Alternatively, an ordered dict of modules can also be passed in.

nn.Sequential是一个有序的容器，该类将按照传入构造器的顺序，依次创建相应的函数，并记录在Sequential类对象的数据结构中，同时以神经网络模块为元素的有序字典也可以作为传入参数。

因此，Sequential是一个有序的容器，该类将按照传入构造器的顺序，依次创建相应的函数，并记录在Sequential类对象的数据结构中，同时以神经网络模块为元素的有序字典也可以作为传入参数。
因此，Sequential可以看成是有多个函数运算对象，串联成的神经网络，其返回的是Module类型的神经网络对象

7.2 以列表的形式，串联函数运算，构建串行执行的神经网络

print("利用系统提供的神经网络模型类:Sequential,以参数列表的方式来实例化神经网络模型对象")

# Example of using Sequential
model_c=nn.Sequential(nn.Linear(28*28,32),nn.ReLU(),nn.Linear(32,10),nn.Softmax(dim=1))
print(model_c)

print("\n显示网络模型参数")
print(model_c.parameters)

print("\n定义神经网络样本输入")
x_input=torch.randn(2,28,28,1)
print(x_input.shape)

print("\n使用神经网络进行预测")
y_pred=model.forward(x.input.size()[0],-1)
print(y_pred)

利用系统提供的神经网络模型类：Sequential,以参数列表的方式来实例化神经网络模型对象
Sequential(
  (0): Linear(in_features=784, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=10, bias=True)
  (3): Softmax(dim=1)
)

显示网络模型参数
<bound method Module.parameters of Sequential(
  (0): Linear(in_features=784, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=10, bias=True)
  (3): Softmax(dim=1)
)>

定义神经网络样本输入
torch.Size([2, 28, 28, 1])

使用神经网络进行预测
tensor([[-0.1526,  0.0437, -0.1685,  0.0034, -0.0675,  0.0423,  0.2807,  0.0527,
         -0.1710,  0.0668],
        [-0.1820,  0.0860,  0.0174,  0.0883,  0.2046, -0.1609,  0.0165, -0.2392,
         -0.2348,  0.1697]], grad_fn=<AddmmBackward>)

# 7.3以字典的形式，串联函数运算，构建串行执行的神经网络
# Example of using Sequential with OrderDict
print("利用系统提供的神经网络模型类：Sequential,以字典的方式实例化神经网络模型对象")
model=nn.Sequential(OrderDict([('h1',nn.Linear(28*28,32)),
                                ('relu1',nn.ReLU()),
                                ('out',nn.Linear(32,10)),
                                ('softmax',nn.Softmax(dim=1))
]))

print(model)

print("\n显示网络模型参数")
print(model.parameters)

print("\n定义神经网络样本输入")
x_input=torch.randn(2,28,28,1)
print(x_input.shape)

print("\n使用神经网络进行预测")
y_pred=model.forward(x_input.size()[0],-1)
print(y_pred)

利用系统提供的神经网络模型类：Sequential,以字典的方式来实例化神经网络模型对象
Sequential(
  (h1): Linear(in_features=784, out_features=32, bias=True)
  (relu1): ReLU()
  (out): Linear(in_features=32, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)

显示网络模型参数
<bound method Module.parameters of Sequential(
  (h1): Linear(in_features=784, out_features=32, bias=True)
  (relu1): ReLU()
  (out): Linear(in_features=32, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)>

定义神经网络样本输入
torch.Size([2, 28, 28, 1])

使用神经网络进行预测
tensor([[0.1249, 0.1414, 0.0708, 0.1031, 0.1080, 0.1351, 0.0859, 0.0947, 0.0753,
         0.0607],
        [0.0982, 0.1102, 0.0929, 0.0855, 0.0848, 0.1076, 0.1077, 0.0949, 0.1153,
         0.1029]], grad_fn=<SoftmaxBackward>)

# 7.4 案例2
nn.Sequential
# Example of using Sequential
modle=nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.Conv2d(20,64,5),
    nn.ReLU()
)

# Example of using Sequential with OrderedDict
model=nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(1,20,5)),
    ('relu1',nn.ReLU()),
    ('conv2',nn.Conv2d(20,64,5))，
    ('relu2',nn.ReLU())
]))

第八章 自定义神经网络模型类（继承于Module类）
# 定义网络模型：带relu的两层全连接神经网络
print("自定义新的神经网络模型的类")

class NetC(torch.nn.Module):
    #定义神经网络
    def __init__(self,n_feature,n_hidden,n_output):
        super(NetC,self).__init__()
        self.h1=nn.Linear(n_feature,n_hidden)
        self.relu1=nn.ReLU()
        self.out=nn.Linear(n_hidden,n_output)
        self.softmax=nn.Softmax(dim=1)
    
    #定义前向运算
    def forward(self,x):
        #得到的数据格式torch.Size([64,1,28,28]) 需要转变为(64,784)
        x=x.view(x.size()[0],-1) # -1表示自动匹配
        h1=self.h1(x)
        a1=self.relu1(h1)
        out=self.out(a1)
        a_out=self.softmax(out)
        return out
print("\n实例化神经网络模型对象")
model=NetC(28*28,32,10)
print(model)

print("\n显示网络模型参数")
print(model.parameters)

print("\n定义神经网络样本输入")
x_input=torch.randn(2,28,28,1)
print(x_input.shape)

print("\n使用神经网络进行预测")
y_pred=model.forward(x_input)
print(y_pred)

自定义新的神经网络模型的类

实例化神经网络模型对象
NetC(
  (h1): Linear(in_features=784, out_features=32, bias=True)
  (relu1): ReLU()
  (out): Linear(in_features=32, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)

显示网络模型参数
<bound method Module.parameters of NetC(
  (h1): Linear(in_features=784, out_features=32, bias=True)
  (relu1): ReLU()
  (out): Linear(in_features=32, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)>

定义神经网络样本输入
torch.Size([2, 28, 28, 1])

使用神经网络进行预测
tensor([[-0.3095,  0.3118,  0.3795, -0.2508,  0.1127, -0.2721, -0.3563,  0.3058,
          0.5193,  0.0436],
        [-0.2070,  0.6417,  0.2125, -0.0413,  0.0827,  0.2448, -0.0399,  0.2837,
          0.0052, -0.1834]], grad_fn=<AddmmBackward>)


# 2、pytorch教程之nn.Module类详解——使用Module类来自定义模型
https://blog.csdn.net/qq_27825451/article/details/90550890

import torch
import torch.nn

class MyNet(nn.Module):
      def __init__(self):
        super(MyNet,self).__init__()  # 调用父类的构造函数
        self.conv1=torch.nn.Conv2d(3,32,3,1,1)
        self.relu1=nn.ReLU()
        self.maxpooling1=nn.MaxPool2d(2,1)
      
        self.conv2=nn.Conv2d(3,32,3,1,1)
        self.relu2=nn.ReLU()
        self.maxpooling2=nn.MaxPool2d(2,1)

        self.dense1=nn.Linear(32*3*3,128)
        self.dense2=nn.Linear(128,10)
      
      def forward(self,x):
          x=self.conv1(x)
          x=self.relu1(x)
          x=self.max_pooling1(x)
          x=self.conv2(x)
          x=self.relu2(x)
          x=self.max_pooling2(x)
          x=self.dense1(x)
          x=self.dense(x)
          return x

model=MyNet()
print(model)


'''运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (max_pooling1): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2): ReLU()
  (max_pooling2): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)

注意：上面的是将所有的层都放在了构造函数__init__里面，但是只是定义了一系列的层，各个层之间到底是什么连接关系并没有，
而是在forward里面实现所有层的连接关系，当然这里依然是顺序连接的。下面再来看一下一个例子：

import torch
import torch.nn.functional as F

class MyNet(torch.nn.Module):
      def __init__(self):
          super(MyNet,self).__init__()  # 第一句话，调用父类的构造函数
          self.conv1=torch.nn.Conv2d(3,32,3,1,1)
          slef.conv2=torch.nn.Conv2d(3,32,3,1,1)

          self.dense1=torch.nn.Linear(32*3*3,128)
          self.dense2=torch.nn.Linear(128,10)

      def forward(self,x):
          x=self.conv1(x)
          x=F.relu(x)
          x=F.max_pool2d(x)
          x=self.conv2(x)
          x=F.relu(x)
          x=F.max_pool2d(x)
          x=self.dense1(x)
          x=self.dense2(x)

model=MyNet()
print(model)

运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)

注意：此时，将没有训练参数的层没有放在构造函数里面了，所以这些层就不会出现在model里面，但是运行关系是在forward里面通过functional的方法实现的。

总结：所有放在构造函数__init__里面的层的都是这个模型的“固有属性”.

# 三、torch.nn.Module类的多种实现
上面是为了一个简单的演示，但是Module类是非常灵活的，可以有很多灵活的实现方式，下面将一一介绍

3.1 通过Sequential来包装层
将几个层包装在一起作为一个大的层（块），前面的一篇文章详细介绍了Sequential类的使用，包括常见的三种方式，以及每一种方式的优缺点，参见：https://blog.csdn.net/qq_27825451/article/details/90551513

所以这里对层的包装当然也可以通过这三种方式了。
（1）方式一：
import torch.nn as nn
from collections import OrderDict
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv_block=nn.Sequential(
          nn.Conv2d(3,32,3,1,1),
          nn.ReLU(),
          nn.MaxPool2d(2))

        self.dense_block=nn.Sequential(
          nn.Linear(32*3*3，128),
          nn.ReLU(),
          nn.Linear(128,10)
        )
      
    #在这里实现层之间的连接关系，其实就是所谓的前向传播

    def forward(self,x):
        conv_out=self.conv_block(x)
        res=conv_out.view(conv_out.size(0),-1)
        out=self.dense_block(res)
        return out

model=MyNet()
print(model)

'''运行结果为：
MyNet(
  (conv_block): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (0): Linear(in_features=288, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=10, bias=True)
  )
)
同前面的文章，这里在每一个包装块里面，各个层是没有名称的，默认按照0、1、2、3、4来排名。

(2)方式二：

import torch.nn as nn
from collections import OrderedDict
class MyNet(nn.Module):
      def __init__(self):
          super(MyNet,self).__init__()
          self.conv_block=nn.Sequential(
            OrderDict(
              [
                ("conv1",nn.Conv2d(3,32,3,1,1)),
                ("relu1",nn.ReLU()),
                ("pool",nn.MaxPool2d(2))

              ]))
          self.dense_block=nn.Sequential(
            OrderDict(
              [
                ("dense1",nn.Linear(32*3*3),128),
                ("relu2",nn.ReLU()),
                ("dense2",nn.Linear(128,10))
              ]))
      def forward(self,x):
          conv_out=self.conv_block(x)
          res=conv_out.view(conv_out.size(0),-1)
          out=self.dense_block(res)
          return out


model=MyNet()
print(model)

'''运行结果为：
MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=288, out_features=128, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=128, out_features=10, bias=True)
  )
)

（3）方式三：
import torch.nn as nn
from collections import OrderedDict
class MyNet(nn.Module):
      def __init__(self):
          super(MyNet,self).__init__()
          self.conv_block=torch.nn.Sequential()
          self.conv_block.add_module('conv1',torch.nn.Conv2d(3,32,3,1,1))
          self.conv_block.add_module('relu1',torch.nn.ReLU())
          self.conv_block.add_module('pool1',torch.nn.MaxPool2d(2))

          self.dense_block=torch.nn.Sequential()
          self.dense_block.add_module('dense1',torch.nn.Linear(32*3*3,128))
          self.dense_block.add_module("relu2",torch.nn.ReLU())
          self.dense_block.add_module("dense2",torch.nn.Linear(128,10))

      def forward(self,x):
          conv_out=self.conv_block(x)
          res=conv_out.view(conv_out.size(0),-1)
          out=self.dense_block(res)
          return out

model=MyNet()
print(model)

'''运行结果为：
MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=288, out_features=128, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=128, out_features=10, bias=True)
  )
)

上面的方式二和方式三，在每一个包装块里面，每个层都是有名称的

3.2 Module类的几个常用方法使用
特别注意：Sequential类虽然继承自Module类，二者有相似部分，但是也有很多不同的部分，集中体现在：
Sequential类实现了整数索引，故而可以使用model[index]这样的方式获取一个层，但是Module类并没有实现整数索引，不能够通过整数索引来获得层，那该怎么办呢？它提供了几个主要的方法，如下：

def children(self):

def named_children(self):

def modules(self):

def named_modules(self,memo=None,prefix='')

这几个方法返回的都是一个Iterator迭代器，故而通过for循环访问，当然也可以通过next

下面就以上面的构建的网络为例子来说明

（1）model.children()方法
import torch.nn as nn
from collections import OrderedDict

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block=torch.nn.Sequential()
        self.conv_block.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv_block.add_module("relu1",torch.nn.ReLU())
        self.conv_block.add_module("pool1",torch.nn.MaxPool2d(2))
 
        self.dense_block = torch.nn.Sequential()
        self.dense_block.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense_block.add_module("relu2",torch.nn.ReLU())
        self.dense_block.add_module("dense2",torch.nn.Linear(128, 10))
 
    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out

model=MyNet()
for i in model.children():
    print(i)
    print(type(i))


'''运行结果为：
Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
<class 'torch.nn.modules.container.Sequential'>
Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
<class 'torch.nn.modules.container.Sequential'>

(2)model.named_children()方法
for i in model.named_children():
    print(i)
    print(type(i))


'''运行结果为：
('conv_block', Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
))
<class 'tuple'>
('dense_block', Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
))
<class 'tuple'>

总结：
（1）model.children()和model.named_children()方法返回的是迭代器iterator;
(2)model.children():每一次迭代返回的每一个元素实际上是Sequential类型，而Sequential类型又可以使用下标index索引来获取每一个Sequential里面的具体层,比如conv层、dense层等；
（3）model.named_children():每一次迭代返回的每一个元素实际上是一个元组类型，元组的第一个元素是名称，第二个元素就是对应的层或者是Sequential。

(3)model.modules()方法
for i in model.modules():
    print(i)
    print("=========================================")

'''运行结果为：
MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=288, out_features=128, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=128, out_features=10, bias=True)
  )
)
==================================================
Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
==================================================
Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
==================================================
ReLU()
==================================================
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
==================================================
Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
==================================================
Linear(in_features=288, out_features=128, bias=True)
==================================================
ReLU()
==================================================
Linear(in_features=128, out_features=10, bias=True)
==================================================

(4) model.named_modules()方法
for i in model.named_modules():
    print(i)
    print("=========================================")

'''运行结果是：
('', MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=288, out_features=128, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=128, out_features=10, bias=True)
  )
))
==================================================
('conv_block', Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
))
==================================================
('conv_block.conv1', Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
==================================================
('conv_block.relu1', ReLU())
==================================================
('conv_block.pool1', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
==================================================
('dense_block', Sequential(
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=128, out_features=10, bias=True)
))
==================================================
('dense_block.dense1', Linear(in_features=288, out_features=128, bias=True))
==================================================
('dense_block.relu2', ReLU())
==================================================
('dense_block.dense2', Linear(in_features=128, out_features=10, bias=True))
==================================================

总结：
（1）model.modules()和model.named_modules()方法返回的是迭代器iterator;
(2)model的modules()方法和named_modules()方法都会将整个模型的所有构成(包括包装层、单独的层、自定义层等)由浅入深依次遍历出来，只不过modules()返回的每一个元素是直接返回的层对象本身，而named_modules()返回的每一个元素是一个元组，第一个元素是名称，第二个元素才是层对象本身。
（3）如何理解children和modules之间的这种差异性。注意pytorch里面不管是模型、层、激活函数、损失函数都可以当成Module的拓展，所以modules和named_modules会层层迭代，由浅入深，将每一个自定义块block、然后block里面的每一个层都当成是modules来迭代。而children就比较直观，就表示的是所谓的"孩子"，所以没有层层迭代深入。

注意：上面这四个方法是以层包装为例来说明的，如果没有层的包装，我们依然可以使用这四个方法，其实结果也是类似的这样去推，这里就不再列出来了。

# pytorch教程之nn.Module类详解——使用Module类来自定义模型
https://blog.csdn.net/qq_27825451/article/details/90550890
前言：pytorch中对于一般的序列模型，直接使用torch.nn.Sequential类可以实现，但是更多的时候面对复杂的模型，比如：多输入多输出、多分支模型、跨层连接模型、带有自定义层的模型等，就需要自己来定义一个模型了。
一、torch.nn.Module类概述
pytorch不像tensorflow那么底层，也不像keras那么高层，这里先比较keras和pytorch的一些小区别。
（1）keras更常见的操作是通过继承Layer类来实现自定义层，不推荐去继承Model类定义模型
(2)pytorch中其实一般没有特别明显的Layer和Moduel的区别，不管是自定义层、自定义块、自定义模型，都是通过继承Module类完成的，这一点很重要。其实Sequential类也是继承自Module类的。
注意：我们当然也可以直接通过继承torch.autograd.Function类来自定义一个层，但是这很不推荐，不提倡，初步原因是需要自己管理parameter
总结：pytorch里面一切自定义操作基本上都是继承nn.Module类来实现的

二、torch.nn.Module类的简介
先来简单看一下它的定义：
class Module(object):
      def __init__(self):
      def forward(self,*input):

      def add_module(self,name,module):
      def cuda(self,device=None):
      def cpu(self):
      def __call__(self,*input,**kwargs):
      def parameters(self,recurse=True):
      def named_parameters(self,prefix='',recurse=True):
      def children(self):
      def named_children(self):
      def modules(self):
      def named_modules(self,memo=None,prefix=''):
      def train(self,mode=True):
      def eval(self):
      def zero_grad(self):
      def __repr__(self):
      def __dir__(self):

还有很多没列出来
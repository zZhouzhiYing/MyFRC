from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from thop import profile
from thop import clever_format
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.avgpool = nn.AvgPool2d(56)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(256, 10)
        self.relu = nn.ReLU(inplace=True)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
    
    # def union():
    #     return self._upsample_add(p5, self.latlayer1(c4))

    def forward(self, x):
        # Bottom-up
        x=self.conv1(x)
        x=self.bn1(x)
        c1 = self.relu(x)
        # c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        print('p5',p5.shape)#([16, 256, 19, 19])
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        print('p2',p2.shape)

        # Smooth
        p4 = self.smooth1(p4)
        print('p4',p4.shape)#([16, 256, 38, 38])
        p3 = self.smooth2(p3)
        print('p3',p3.shape)#[16, 256, 75, 75]
        p2 = self.smooth3(p2)
        print('p2',p2.shape)#[16, 256, 150, 150]

        #become classfication
        # x = self.avgpool(p2)
        # print('av',x.shape)
        # x = x.view(x.size(0), -1)
        # print('x',x.size())
        # x = self.fc(x)
        # print('x',x.size())
        # return x
        return p2, p3, p4, p5
        
class F1(FPN,nn.Module):
    def __init__(self,block,layers):
        super(F1, self).__init__(block,layers)
        model=FPN(Bottleneck, [3, 4, 6, 3])
        self.conv1=model.conv1
        self.bn1=model.bn1
        self.relu=model.relu
        self.layer1=model.layer1
        self.layer2=model.layer2
        self.layer3=model.layer3
        self.layer4=model.layer4
        self.toplayer=model.toplayer
        self.smooth1=model.smooth1
        self.smooth2=model.smooth2
        self.smooth3=model.smooth3
        self.latlayer1=model.latlayer1
        self.latlayer2=model.latlayer2
        self.latlayer3=model.latlayer3
        
        
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        c1 = self.relu(x)
        # c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        print('p5',p5.shape)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        print('p4',p4.shape)
        p3 = self.smooth2(p3)
        print('p3',p3.shape)
        # p2 = self.smooth3(p2)
        # print('p2',p2.shape)

        return p4

class F2(FPN,nn.Module):
    def __init__(self,block,layers):
        super(F2, self).__init__(block,layers)
        model=FPN(Bottleneck, [3, 4, 6, 3])
        # self.conv1=model.conv1
        # self.bn1=model.bn1
        # self.relu=model.relu
        # self.maxpool=model.maxpool
        # self.layer1=model.layer1
        # self.layer2=model.layer2
        # self.layer3=model.layer3
        self.layer4=model.layer4
        self.avgpool=model.avgpool
        self.fc=model.fc
    def forward(self, x):
        x = self.smooth3(x)
        # x = self.layer4(x)
        # print('l4',x.shape)
        # x = self.avgpool(x)
        # print('heihei')
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x
def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])
def Fpn50():
    model = FPN(Bottleneck, [3, 4, 6, 3])
    one=F1(Bottleneck, [3, 4, 6, 3])
    cla=F2(Bottleneck, [3, 4, 6, 3])
    # 获取特征提取部分
    
    features = list([one])
    # 获取分类部分
    classifier = list([cla])
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)

    return features,classifier


# x=torch.randn(16,3,600,600)
 
# net=FPN50()
# output=net(x)

# macs, params = profile(net, inputs=(x,))
# macs, params = clever_format([macs, params], "%.3f")
# print(params)
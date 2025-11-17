import os
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


# ------------------ 自定义层定义 ------------------
class SeparableConv2DBlock(nn.Module):
    """
    等效于 Keras 的 SeparableConv2D。
    先做深度可分卷积(depthwise)，再做 pointwise 卷积。
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        # Depthwise：对每个输入通道分别做卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 深度可分
            bias=False  # 一般 depthwise 不加 bias
        )
        # Pointwise：再把通道数从 in_channels 压/升到 out_channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ------------------ （示例）局部卷积的简单实现 ------------------ #


class TorchLocallyConnected2D(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(TorchLocallyConnected2D, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

    
# ------------------ PyTorch 模型结构 ------------------ #
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ========== Block 1 ==========
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # → [B,64,48,48]
        # self.bn1 = nn.BatchNorm2d(64,moment)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-3)
        self.sigmoid1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.1) 

        # ========== Block 2 ==========
        self.conv2 = nn.Conv2d(64, 100, kernel_size=3, stride=1)  # → [B,100,46,46]
        self.bn2 = nn.BatchNorm2d(100,momentum=0.01, eps=1e-3)
        self.sigmoid2 = nn.Sigmoid()
        # 注意：Keras 这里 SeparableConv2D(100,(3,3)), strides=1, 没有 padding
        # 输出形状从 (46,46) -> (44,44)
        self.sep_conv1 = SeparableConv2DBlock(100, 100, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)  # → [B,100,22,22]
        self.bn3 = nn.BatchNorm2d(100,momentum=0.01, eps=1e-3)
        self.sigmoid3 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(0.1)

        # ========== Residual Block 1 ==========
        # 残差支路：Conv2D(100->200, kernel=1)，再 BN
        self.res_conv1 = nn.Conv2d(100, 200, kernel_size=1, stride=1, bias=False)
        self.res_bn1 = nn.BatchNorm2d(200,momentum=0.01, eps=1e-3)

        # 主分支：SeparableConv2D(100->200, padding='same') + BN + 激活 + SeparableConv2D(200->200, same)
        self.sep_conv2 = SeparableConv2DBlock(100, 200, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(200,momentum=0.01, eps=1e-3)
        self.sigmoid4 = nn.Sigmoid()

        self.sep_conv3 = SeparableConv2DBlock(200, 200, kernel_size=3, stride=1, padding=1)
        self.sigmoid5 = nn.Sigmoid()
        self.dropout3 = nn.Dropout(0.1)

        # ========== Residual Block 2 ==========
        # 残差支路：Conv2D(200->400, kernel=1)，再 BN
        self.res_conv2 = nn.Conv2d(200, 400, kernel_size=1, stride=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(400,momentum=0.01, eps=1e-3)

        # 主分支：SeparableConv2D(200->400, same) + BN + 激活 + SeparableConv2D(400->400, same)
        self.sep_conv4 = SeparableConv2DBlock(200, 400, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(400,momentum=0.01, eps=1e-3)
        self.sigmoid6 = nn.Sigmoid()

        self.sep_conv5 = SeparableConv2DBlock(400, 400, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)  # 输出 [B,400,11,11]
        self.sigmoid7 = nn.Sigmoid()
        self.dropout4 = nn.Dropout(0.1)

        # ========== Residual Block 3 ==========
        # 残差支路：直接 residual = x
        self.sep_conv6 = SeparableConv2DBlock(400, 400, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(400,momentum=0.01, eps=1e-3)
        self.sigmoid8 = nn.Sigmoid()

        self.sep_conv7 = SeparableConv2DBlock(400, 400, kernel_size=3, stride=1, padding=1)
        self.sigmoid9 = nn.Sigmoid()
        self.dropout5 = nn.Dropout(0.1)

        # Cropping2D((2,2)) 相当于在 [H,W] 方向各裁剪2
        # 由于上一层输出是 [B,400,11,11]，裁剪后变成 [B,400,7,7]
        self.crop = lambda x: x[:, :, 2:-2, 2:-2]

        # ========== 后续层 ==========
        # Keras reshape → (B,49,20,20)；再 permute → (B,20,20,49)（channels-last）。
        # 在 PyTorch 中一般使用 channels-first，所以可以省略 permute，只要保证下游层 in_channels=49 即可。
        self.conv3 = nn.Conv2d(49, 32, kernel_size=3, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(32,momentum=0.01, eps=1e-3)
        self.sigmoid10 = nn.Sigmoid()

        # LocallyConnected2D(16,(3,3))；这里是简化版
        self.local_conv1 = TorchLocallyConnected2D(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            # padding=0,
            output_size=(16, 16),  # 根据前层输出计算得到,
            bias=True
        )
        self.sigmoid11 = nn.Sigmoid()

        # Conv2D(16->64, kernel=3, padding='same')
        self.conv4 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)

        # Conv2DTranspose(64->1, (8,8), strides=(8,8))
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=8)

        # LocallyConnected2D(1->1, (1,1))
        self.local_conv2 = TorchLocallyConnected2D(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            stride=1,
            # padding=0,
            output_size=(128, 128),
            bias=True
        )
        # 最后 flatten -> (B, 128*128) = (B,16384)
        # PyTorch 中可用 nn.Flatten() 或手动 x.view
        self.flatten = nn.Flatten()

    def forward(self, x):
        # ---------- Block 1 ----------
        x = self.conv1(x)       # (B,64,96,96)
        x = self.maxpool1(x)    # (B,64,48,48)
        x = self.bn1(x)
        x = self.sigmoid1(x)
        x = self.dropout1(x)

        # ---------- Block 2 ----------
        x = self.conv2(x)       # (B,100,46,46)
        x = self.bn2(x)
        x = self.sigmoid2(x)
        x = self.sep_conv1(x)   # (B,100,44,44)
        x = self.maxpool2(x)    # (B,100,22,22)
        x = self.bn3(x)
        x = self.sigmoid3(x)
        x = self.dropout2(x)

        # ---------- Residual Block 1 ----------
        tmp = x

        x = self.sep_conv2(x)   # (B,200,22,22)
        x = self.bn4(x)
        x = self.sigmoid4(x)

        residual = self.res_conv1(tmp)  # (B,200,22,22)
        x = self.sep_conv3(x)   # (B,200,22,22)
        residual = self.res_bn1(residual)
        # 主分支
        x = x + residual
        x = self.sigmoid5(x)
        x = self.dropout3(x)

        # ---------- Residual Block 2 ----------
        tmp = x
        x = self.sep_conv4(x)   # (B,400,22,22)
        x = self.bn5(x)
        x = self.sigmoid6(x)

        residual = self.res_conv2(tmp)  # (B,400,22,22)

        x = self.sep_conv5(x)   # (B,400,22,22)
        residual = self.res_bn2(residual)

        x = x + residual
        x = self.maxpool3(x)    # (B,400,11,11)
        x = self.sigmoid7(x)
        x = self.dropout4(x)

        # ---------- Residual Block 3 ----------
        residual = x            # (B,400,11,11)
        x = self.sep_conv6(x)   # (B,400,11,11)
        x = self.bn6(x)
        x = self.sigmoid8(x)
        x = self.sep_conv7(x)   # (B,400,11,11)

        x = x + residual
        x = self.sigmoid9(x)
        x = self.dropout5(x)

        # Crop => (B,400,7,7)·  
        x = self.crop(x)

        # Keras 做了 Reshape((49,20,20)) + Permute((2,3,1)) => (20,20,49)
        # 在 PyTorch，我们直接保留 channels-first，不再 permute。
        # 因此输入 conv3 时 in_channels=49 即可。
        x = x.permute(0, 2, 3, 1)  # [B,7,7,400]
        B, H, W, C = x.shape
        x = x.reshape(B, 49, 20, 20)  # 分解为 (49,20,20)，模拟TF的Reshape
        
        # 调整回PyTorch的通道优先，并确保维度正确
        # x = x.permute(0, 3, 1, 2)  # [B,20,20,49] → [B,49,20,20]（正确通道优先）
        
        x = self.conv3(x)       # [B,32, 18,18]
        x = self.bn7(x)
        x = self.sigmoid10(x)

        # LocallyConnected2D(16,(3,3)) => [B,16,16,16]（简化）
        x = self.local_conv1(x)
        x = self.sigmoid11(x)

        # Conv2D(16->64, kernel=3, padding='same') => [B,64,16,16]
        x = self.conv4(x)

        # Deconv => (64->1, kernel=8, stride=8) => [B,1,128,128]
        x = self.deconv(x)

        # LocallyConnected2D(1->1, (1,1)) => (B,1,128,128)
        x = self.local_conv2(x)
        print(x.shape)

        # Flatten => (B, 1*128*128)= (B,16384)
        x = x.permute(0, 2, 3, 1)  # PyTorch 改成 Channels-Last
        x = self.flatten(x)
        return x


if __name__ == "__main__":
    neuron_root = r'lineage_1\gen_1_folder\imgset\gen_1_image_shade'
    img_all = []
    for neuron in os.listdir(neuron_root):
        neuron_path = os.path.join(neuron_root,neuron)
        # print(neuron_path)
        mat = scio.loadmat(neuron_path)
        mat = mat['imgset']
        mat = np.array(mat)
        img_all.append(mat)
    img_all = np.array(img_all)
    print(img_all.shape)
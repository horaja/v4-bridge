import torch
import torch.nn as nn
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
from torch.nn.modules.utils import _pair
# import matplotlib.pyplot as plt

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

# ------------------ 权重加载工具函数 ------------------
def load_weights_from_tf(tf_model, pt_model):
    tf_weights = tf_model.get_weights()
    ptr = 0  # 全局指针
    
    # ================== 调试工具 ==================
    def print_layer_status(layer_type, tf_shape, pt_shape):
        status = "✅" if tf_shape == pt_shape else "❌"
        print(f"[{ptr:03d}] {layer_type:25} | TF: {str(tf_shape):20} → PT: {str(pt_shape):20} {status}")

    # ================== 核心加载方法 ==================
    def _load_conv2d(pt_layer, has_bias=True):
        nonlocal ptr
        tf_w = tf_weights[ptr]
        
        # 特殊处理1x1卷积 (H,W,in,out) = (1,1,100,200)
        if pt_layer.kernel_size == (1,1):
            # 转换维度顺序 (1,1,100,200) → (200,100,1,1)
            pt_w = torch.from_numpy(tf_w.transpose(3, 2, 0, 1)).float()
        else:
            pt_w = torch.from_numpy(tf_w.transpose(3, 2, 0, 1)).float()
        
        print(f"[{ptr}] Conv2D {tuple(pt_layer.kernel_size)} | TF: {tf_w.shape} → PT: {pt_w.shape}")
        
        pt_layer.weight.data = pt_w
        ptr += 1

        if has_bias and pt_layer.bias is not None:
            pt_layer.bias.data = torch.from_numpy(tf_weights[ptr]).float()
            ptr += 1

    def _load_separable_conv2d(pt_layer, has_bias=True):
        nonlocal ptr
        try:
            # 加载深度卷积权重（无偏置）
            tf_depthwise = tf_weights[ptr]
            print_layer_status("SeparableConv2D", tf_depthwise.shape, pt_layer.depthwise.weight.shape)
            pt_depthwise = torch.from_numpy(tf_depthwise.transpose(2, 3, 0, 1)).float()  # (in,1,H,W)
            pt_layer.depthwise.weight.data = pt_depthwise
            ptr += 1

            # 加载逐点卷积权重
            tf_pointwise = tf_weights[ptr]
            pt_pointwise = torch.from_numpy(tf_pointwise.transpose(3, 2, 0, 1)).float()  # (out,in,1,1)
            pt_layer.pointwise.weight.data = pt_pointwise
            ptr += 1

            # 加载偏置项（如果存在）
            if has_bias:
                tf_bias = tf_weights[ptr]
                pt_layer.pointwise.bias.data = torch.from_numpy(tf_bias).float()  # 关键修正：加载到逐点卷积的偏置
                ptr += 1
                
            print(f"[{ptr}] SeparableConv2D加载完成 | 深度卷积: {tf_depthwise.shape} | 逐点卷积: {tf_pointwise.shape} | 偏置: {tf_bias.shape if has_bias else '无'}")

        except Exception as e:
            print(f"加载SeparableConv2D失败: {str(e)}")
            raise

    def _load_batchnorm(pt_layer):
        nonlocal ptr
        gamma = torch.from_numpy(tf_weights[ptr]).float()
        beta = torch.from_numpy(tf_weights[ptr+1]).float()
        mean = torch.from_numpy(tf_weights[ptr+2]).float()
        var = torch.from_numpy(tf_weights[ptr+3]).float()
        print_layer_status("BatchNorm", tuple(w.shape for w in tf_weights[ptr:ptr+4]), (gamma.shape, beta.shape, mean.shape, var.shape))
        pt_layer.weight.data = gamma
        pt_layer.bias.data = beta
        pt_layer.running_mean.data = mean
        pt_layer.running_var.data = var
        # pt_layer.track_running_stats = False  # 强制使用预训练的均值和方差
        ptr += 4

    def _load_local_conv(pt_layer, expected_output_size):
        nonlocal ptr
        # print(111111111111111111111111111111111111)
        tf_w = tf_weights[ptr]
        H_out, W_out = expected_output_size
        kernel_size = (pt_layer.kernel_size, pt_layer.kernel_size) if isinstance(pt_layer.kernel_size, int) else pt_layer.kernel_size
        in_k = kernel_size[0] * kernel_size[1]  # 计算输入核参数
        in_ch = pt_layer.in_channels  

        # ================== 处理主权重 ==================
        # if tf_w.ndim == 3:  # Case 1: 三维权重 (Layer 46)
            # -------------------------------------------------------------
            # TensorFlow 权重形状: (H_out*W_out=256, in_k=288, out_ch=16)
            # PyTorch 期望形状:   (out_ch*H_out*W_out=16*16*16=4096, in_k=288)
            # -------------------------------------------------------------
        try:
            # Step 1: 重塑为空间位置格式 (H_out, W_out, in_k, out_ch)
            tf_w = tf_w.reshape(H_out, W_out, in_k, in_ch,-1)  # (16,16,9,32,16)
            
            # # Step 2: 交换 H 和 W 维度（匹配 PyTorch 列优先顺序）
            tf_w = tf_w.transpose(4,3,0,1,2)  # (16,16,32,9,16) → (16,32,16,16,9)
            
            # # Step 3: 合并输出通道和空间维度
            pt_w = tf.expand_dims(tf_w, axis=0).numpy() 
            
            # # Step 4: 转置以匹配 PyTorch 的线性层形状 (out_ch*H_out*W_out, in_k)
            # pt_w = pt_w.transpose(1, 0) if pt_layer.weight.shape[1] != in_k else pt_w  # 关键判断
            
            # 验证形状
            assert pt_w.shape == pt_layer.weight.shape, \
                f"权重形状不匹配！预期 {pt_layer.weight.shape}，实际 {pt_w.shape}"
                
        except Exception as e:
            raise ValueError(f"加载三维权重失败: {str(e)}")

        # 加载权重到 PyTorch
        pt_layer.weight.data = torch.from_numpy(pt_w).float()
        ptr += 1
        print(f"[{ptr}] LocalConv2D {pt_layer.kernel_size} | TF: {tf_w.shape} → PT: {pt_w.shape}")

        # ================== 处理偏置 ==================

        tf_b = tf_weights[ptr]
        # print(tf_b.shape)
        
        # TensorFlow 形状: (H_out=16, W_out=16, out_ch=16)
        # PyTorch 期望:    (out_ch=16, H_out=16, W_out=16)
        pt_b = tf_b.transpose(2, 0, 1)  # (16,16,16) → (16,16,16)
        # pt_b = pt_b.reshape(-1)  # 展平为 (16*16*16)
        pt_b = np.expand_dims(pt_b, axis=0)  # (16*16*16) → (1,16*16*16)
    
        # # 验证偏置形状
        # assert pt_b.size == pt_layer.bias.numel(), \
        #     f"偏置数量不匹配！预期 {pt_layer.bias}，实际 {pt_b.size}"
        
        pt_layer.bias.data = torch.from_numpy(pt_b).float()
        ptr += 1
            



    # ================== 按实际顺序加载 ==================
    try:
        # Block 1
        _load_conv2d(pt_model.conv1)            # Layer1 (ptr=0)
        _load_batchnorm(pt_model.bn1)           # Layer3 (ptr=2)

        # Block 2
        _load_conv2d(pt_model.conv2)            # Layer6 (ptr=6)
        _load_batchnorm(pt_model.bn2)           # Layer7 (ptr=8)
        _load_separable_conv2d(pt_model.sep_conv1, has_bias=True)  # Layer9 (ptr=12)
        _load_batchnorm(pt_model.bn3)           # Layer11 (ptr=15)

        # ------------------ Residual Block 1 ------------------
        # 主路径：SeparableConv2D_1 (对应Layer14)
        _load_separable_conv2d(pt_model.sep_conv2, has_bias=True)  # ptr=19 → 加载3个权重
        _load_batchnorm(pt_model.bn4)                              # ptr=22 → 加载4个参数

        # 残差路径: Layer17 (res_conv1)
        _load_conv2d(pt_model.res_conv1, has_bias=False)           # ptr=26 → 加载1个权重
        _load_separable_conv2d(pt_model.sep_conv3, has_bias=True)  # ptr=30 → 加载3个权重

        _load_batchnorm(pt_model.res_bn1)                          # ptr=26 → 加载4个参数

      
        # ------------------ Residual Block 2 ------------------
        # 残差路径 (先加载)
        _load_separable_conv2d(pt_model.sep_conv4, has_bias=True)  # ptr=44 (depth=3x3x200x1, point=1x1x200x400)
        _load_batchnorm(pt_model.bn5)                      # ptr=47 (4个400参数)

        _load_conv2d(pt_model.res_conv2, has_bias=False)   # ptr=36 (1x1x200x400)
        _load_separable_conv2d(pt_model.sep_conv5, has_bias=True)  # ptr=51 (depth=3x3x400x1, point=1x1x400x400)
        _load_batchnorm(pt_model.res_bn2)                  # ptr=40 (4个400参数)
        
        
        # ------------------ Residual Block 3 ------------------
        _load_separable_conv2d(pt_model.sep_conv6, has_bias=True)  # ptr=54 (depth=3x3x400x1, point=1x1x400x400)
        _load_batchnorm(pt_model.bn6)                      # ptr=57 (4个400参数)
        _load_separable_conv2d(pt_model.sep_conv7, has_bias=True)  # ptr=61 (depth=3x3x400x1, point=1x1x400x400)

        # 后续层
        _load_conv2d(pt_model.conv3)
        _load_batchnorm(pt_model.bn7)

        _load_local_conv(pt_model.local_conv1,expected_output_size=(16, 16)  # 根据网络结构确定
        )
        _load_conv2d(pt_model.conv4)
        _load_conv2d(pt_model.deconv)
        _load_local_conv(pt_model.local_conv2, expected_output_size=(128, 128))

        # 最终验证
        if ptr != len(tf_weights):
            raise ValueError(
                f"权重未完全加载！剩余 {len(tf_weights)-ptr} 个权重\n"
                f"最后处理的层：{pt_model._modules[list(pt_model._modules.keys())[-1]]}"
            )

        return pt_model

    except Exception as e:
        print(f"\n{'!'*40}")
        print(f"加载失败于ptr={ptr} ({tf_weights[ptr].shape if ptr<len(tf_weights) else 'EOF'})")
        print(f"错误类型：{type(e).__name__}")
        print(f"错误详情：{str(e)}")
        print(f"当前PyTorch层：{pt_model._modules[list(pt_model._modules.keys())[ptr//2]]}")
        print(f"{'!'*40}")
        raise

class CustomModelBuilder:
    def __init__(self):
        self._fix_legacy_config()

    def _fix_legacy_config(self):
        """解决旧版TensorFlow配置问题"""
        if hasattr(tf.compat, 'v1'):
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.compat.v1.InteractiveSession(config=config)

    def build_model(self):
        """构建完整模型结构"""
        # ------------------ 输入层 ------------------
        image_input = Input(shape=(100,100,3))
        
        # ------------------ 特征提取部分 ------------------
        x = self._build_feature_extractor(image_input)
        
        # ------------------ 输出层 ------------------
        x = self._build_output_layers(x)
        
        # ------------------ 完整模型 ------------------
        model = Model(inputs=image_input, outputs=x)
        
        # ------------------ 特殊权重初始化 ------------------
        self._custom_weights_init(model)
        
        return model

    def _build_feature_extractor(self, x):
        """构建特征提取模块"""
        # Block 1
        x = Conv2D(64, (5, 5), strides=(1,1))(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        x = Dropout(0.1)(x)

        # Block 2
        x = Conv2D(100, (3, 3), strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        x = SeparableConv2D(100, (3,3), strides=(1,1))(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        x = Dropout(0.1)(x)

        # Residual Block 1
        residual = Conv2D(200, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(200, (3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        x = SeparableConv2D(200, (3,3), strides=(1,1), padding='same')(x)
        x = Add()([x, residual])
        x = Activation('sigmoid')(x)
        x = Dropout(0.1)(x)

        # Residual Block 2
        residual = Conv2D(400, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(400, (3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        x = SeparableConv2D(400, (3,3), strides=(1,1), padding='same')(x)
        x = Add()([x, residual])
        x = MaxPooling2D((2, 2))(x)
        x = Activation('sigmoid')(x)
        x = Dropout(0.1)(x)

        # Residual Block 3
        residual = x
        x = SeparableConv2D(400, (3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        x = SeparableConv2D(400, (3,3), strides=(1,1), padding='same')(x)
        x = Add()([x, residual])
        x = Activation('sigmoid')(x)
        x = Dropout(0.1)(x)
        x = Cropping2D(cropping=((2, 2), (2, 2)))(x)

        return x

    def _build_output_layers(self, x):
        """构建输出层模块"""
        x = Reshape((49,20,20))(x)
        x = Permute((2, 3, 1))(x)
        x = Conv2D(32, (3, 3), strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        x = LocallyConnected2D(16, (3,3), implementation=1)(x)
        x = Activation('sigmoid')(x)
        x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
        x = Conv2DTranspose(1, (8,8), strides=(8,8))(x)
        x = LocallyConnected2D(1, (1,1), implementation=3)(x)
        return Flatten()(x)

    def _custom_weights_init(self, model):
        """自定义权重初始化（对应原始代码的特殊处理）"""
        # 获取 Conv2DTranspose 层
        transpose_layer = model.layers[-3]
        
        # 初始化特殊权重
        w2 = np.zeros((8, 8, 1, 64))
        ni = 0
        for i in range(8):
            for j in range(8):
                w2[i,j,0,ni] = 1
                ni += 1
        
        # 设置权重并冻结
        transpose_layer.set_weights([w2, np.zeros(1)])
        transpose_layer.trainable = False

    def load_pretrained(self, model, weight_path):
        """加载预训练权重"""
        try:
            model.load_weights(weight_path)
            # 添加在 load_pretrained 方法中
            for i, layer in enumerate(model.layers):
                if layer.get_weights():
                    print(f"Layer {i}: {layer.name}")
                    for w in layer.get_weights():
                        print(f"  Weight shape: {w.shape}")
            print("成功加载预训练权重！")
        except Exception as e:
            print(f"权重加载失败: {str(e)}")
            print("请检查：")
            print("1. 模型结构与权重文件是否匹配")
            print("2. 文件路径是否正确")
            print("3. TensorFlow版本是否兼容")

def compare_results(tf_output, pt_output):
    print("[结果对比]")
    print(f"TensorFlow 输出形状: {tf_output.shape}")
    print(f"PyTorch 输出形状:   {pt_output.shape}")
    
    # 形状对齐检查
    assert tf_output.shape == pt_output.shape, "输出形状不匹配！"
    
    # 数值精度检查
    abs_diff = np.abs(tf_output - pt_output)
    print("\n数值差异统计：")
    print(f"最大绝对差异: {abs_diff.max():.6f}")
    print(f"平均绝对差异: {abs_diff.mean():.6f}")
    print(f"差异超过 1e-3 的比例：{(abs_diff > 1e-3).mean()*100:.2f}%")
    print(abs_diff.mean())
    
    # # 可视化对比
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.hist(tf_output.flatten(), bins=50, alpha=0.5, label='TF')
    # plt.hist(pt_output.flatten(), bins=50, alpha=0.5, label='PT')
    # plt.legend()
    # plt.title("输出分布对比")
    
    # plt.subplot(1, 2, 2)
    # plt.scatter(tf_output.flatten(), pt_output.flatten(), s=1)
    # plt.plot([-3, 3], [-3, 3], 'r--')
    # plt.xlabel("TensorFlow 输出")
    # plt.ylabel("PyTorch 输出")
    # plt.title("样本点对应关系")
    # plt.show()

# 提取TensorFlow中间层输出的工具函数
def build_tf_intermediate_model(tf_model):
    ignored_layers = (tf.keras.layers.InputLayer, tf.keras.layers.Dropout, tf.keras.layers.Lambda, tf.keras.layers.Flatten, 
                      tf.keras.layers.Add, tf.keras.layers.Cropping2D, tf.keras.layers.Reshape, tf.keras.layers.Permute)

    return tf.keras.Model(
        inputs=tf_model.input, 
        outputs=[layer.output for layer in tf_model.layers if not isinstance(layer, ignored_layers)]
    )


# 提取PyTorch中间层输出的Hook
def get_pt_intermediate_outputs(pt_model, input_tensor):
    outputs = {}
    hooks = []

    # 仅 Hook 第一层
    def hook_wrapper(name):
        def hook(module, input, output):
            outputs[name] = output.detach().cpu().numpy()
        return hook

    for name, module in pt_model.named_children():  # 只遍历第一层
        if not isinstance(module, torch.nn.Dropout):  # 过滤 Dropout 层
            hooks.append(module.register_forward_hook(hook_wrapper(name)))

    pt_model.eval()
    with torch.no_grad():
        pt_model(input_tensor)

    for h in hooks:
        h.remove()

    return outputs

def validate_all_layers(tf_model, pt_model, np_input, pt_input):
    print("=== 开始逐层验证 ===")

    # 过滤 TensorFlow 里的 Input, Add, Cropping2D, Reshape, Permute
    tf_intermediate_model = build_tf_intermediate_model(tf_model)
    tf_outputs = tf_intermediate_model.predict(np_input)

    # 获取 PyTorch 所有层的输出（跳过 Dropout）
    pt_outputs = get_pt_intermediate_outputs(pt_model, pt_input)

    tf_layer_names = [layer.name for layer in tf_model.layers]

    # 获取层名称（过滤 Input）
    tf_layer_names = [layer.name for layer in tf_model.layers if not isinstance(layer, 
                      (tf.keras.layers.InputLayer, tf.keras.layers.Dropout, tf.keras.layers.Add, 
                       tf.keras.layers.Cropping2D, tf.keras.layers.Reshape, tf.keras.layers.Permute))]

    pt_layer_names = [name for name in pt_outputs.keys()]

    # 遍历所有层进行对比
    for i, (tf_layer_name, pt_layer_name) in enumerate(zip(tf_layer_names, pt_layer_names)):
        layer = tf_model.get_layer(tf_layer_name)
        print(f"\n=== 验证层 {i}: {tf_layer_name} (TF) <-> {pt_layer_name} (PT) ===")

        # 获取输出
        tf_layer_output = tf_outputs[i]
        print(f"TensorFlow 输出形状: {tf_layer_output.shape}")
        pt_layer_output = pt_outputs[pt_layer_name]
        print(f"PyTorch 输出形状:   {pt_layer_output.shape}")

        # 需要调整 TensorFlow 输出格式为 NCHW
        if tf_layer_output.ndim == 4:
            tf_layer_output_pt_format = tf_layer_output.transpose(0, 3, 1, 2)
        else:
            tf_layer_output_pt_format = tf_layer_output
        
        max_diff = np.abs(tf_layer_output_pt_format - pt_layer_output).max()
        print(f"输出最大差异: {max_diff:.6f}")

        # 允许一定误差范围
        assert np.allclose(tf_layer_output_pt_format, pt_layer_output, atol=1e-5, rtol=1e-3), "输出不匹配"

    print("✅ 所有层验证通过！")


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 初始化构建器
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)
    builder = CustomModelBuilder()
    
    # 构建模型
    model = builder.build_model()
    # 加载预训练权重
    weight_path = r"F:\Lee_Lab_Data\Additional_Data\best_model_MKB.hdf5"
    builder.load_pretrained(model, weight_path)
    
    pt_model = TorchModel()
    pt_model = load_weights_from_tf(model, pt_model)

    # print(pt_model)  # 打印模型结构

    # 保存模型参数（state_dict）
    torch.save(pt_model.state_dict(), 'pytorch_model_weights_MKB.pth')

    # 后续加载方式
    loaded_model = TorchModel()  # 需先实例化模型结构
    loaded_model.load_state_dict(torch.load('pytorch_model_weights_MKB.pth'))
    loaded_model.eval()  # 切换到评估模式

    # TensorFlow输入 (NHWC)
    from PIL import Image
    import torchvision.transforms as T

    # 加载图像并处理为 TensorFlow 输入格式 NHWC
    img_path = r"D:\Desktop\LeeLab\3D_Sti\V4_solid_flat_data_code\photo\lineage_1\gen_0_folder\gen_0_image\match_texture_2d\0\0_matched.png"
    img = Image.open(img_path).convert("RGB")

    # 假设模型输入是 (100, 100, 3)，你可以根据需要 resize
    transform = T.Compose([
        T.Resize((100, 100)),
        T.ToTensor(),                 # → [0,1] float32, shape: (3, H, W)
        # T.Normalize(mean=[0.5]*3, std=[0.5]*3)  # 若模型训练时有用标准化
    ])

    img_tensor = transform(img)  # shape: (3, 100, 100)
    pt_input = img_tensor.unsqueeze(0)  # (1, 3, 100, 100) for PyTorch

    np_input = pt_input.numpy().transpose(0, 2, 3, 1)  # → NHWC for TensorFlow


    # # 验证输入转换正确性
    # def validate_separable_conv():
    # # 获取TensorFlow层权重
    #     tf_depthwise = model.layers[9].get_weights()[0]  # 假设第9层是SeparableConv2D
    #     tf_pointwise = model.layers[9].get_weights()[1]

    #     # 获取PyTorch层权重
    #     pt_depthwise = loaded_model.sep_conv1.depthwise.weight.data.numpy()
    #     pt_pointwise = loaded_model.sep_conv1.pointwise.weight.data.numpy()

    #     # 验证维度转换
    #     assert np.allclose(tf_depthwise.transpose(2,3,0,1), pt_depthwise, atol=1e-6)  # TF→PT维度转换
    #     assert np.allclose(tf_pointwise.transpose(3,2,0,1), pt_pointwise, atol=1e-6)
    #     print
    #     ("SeparableConv2D权重转换正确！")
    
    # validate_separable_conv()
    

    with tf.device('/CPU:0'):
        
        tf_output = model.predict(np_input)
    
    with torch.no_grad():
        pt_output = loaded_model(pt_input).numpy()

    compare_results(tf_output, pt_output)

    # np_input = np.random.randn(1, 100, 100, 3).astype(np.float32)
    # pt_input = torch.from_numpy(np_input.transpose(0, 3, 1, 2)).float()

    
    
    # # 执行逐层验证
    # validate_all_layers(model, loaded_model, np_input, pt_input)
    
    # # 最终输出对比
    # with tf.device('/CPU:0'):
    #     tf_output = model.predict(np_input)
    # with torch.no_grad():
    #     pt_output = loaded_model(pt_input).numpy()
    
    # compare_results(tf_output, pt_output)
    # print("所有层验证通过！")

    
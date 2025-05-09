"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.modules.block import ContextGuidedBlock_Down

from .utils import get_activation

from ..core import register

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    """
    融合卷积与归一化层的模块（支持部署时结构优化）
    包含Conv2d + BatchNorm2d + 激活函数，支持转换为无归一化的部署模式
    """
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: 输入特征张量 [B, in_channels, H, W]
        Returns:
            torch.Tensor: 输出特征张量 [B, out_channels, H, W]
        逻辑说明:
            1. 输入特征通过两个1x1卷积分为主分支和残差分支
            2. 主分支经过num_blocks个瓶颈模块提取特征
            3. 主分支与残差分支相加融合
            4. 通过1x1卷积调整通道数至out_channels
        """
        """
        前向传播（split拆分方式）
        Args:
            x: 输入特征图 [B, c1, H, W]
        Returns:
            torch.Tensor: 融合后的特征图 [B, c2, H, W]
        逻辑说明:
            1. 通过1x1卷积将输入通道扩展至c3
            2. 按通道均分为两个分支（split方式）
            3. 分支1经过CSP层+3x3卷积得到特征A
            4. 分支2经过CSP层+3x3卷积得到特征B
            5. 合并原始分支、特征A、特征B后通过1x1卷积输出
        """
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: 输入特征张量 [B, in_channels, H, W]
        Returns:
            torch.Tensor: 输出特征张量 [B, out_channels, H, W]
        逻辑说明:
            1. 输入特征通过两个1x1卷积分为主分支和残差分支
            2. 主分支经过num_blocks个瓶颈模块提取特征
            3. 主分支与残差分支相加融合
            4. 通过1x1卷积调整通道数至out_channels
        """
        """
        前向传播（split拆分方式）
        Args:
            x: 输入特征图 [B, c1, H, W]
        Returns:
            torch.Tensor: 融合后的特征图 [B, c2, H, W]
        逻辑说明:
            1. 通过1x1卷积将输入通道扩展至c3
            2. 按通道均分为两个分支（split方式）
            3. 分支1经过CSP层+3x3卷积得到特征A
            4. 分支2经过CSP层+3x3卷积得到特征B
            5. 合并原始分支、特征A、特征B后通过1x1卷积输出
        """
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    """
    空间压缩下采样层（Spatial Compression Downsample）
    通过1x1卷积+深度可分离卷积实现高效下采样
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 卷积核尺寸
        s (int): 步长（下采样倍数）
    """
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: 输入特征张量 [B, in_channels, H, W]
        Returns:
            torch.Tensor: 输出特征张量 [B, out_channels, H, W]
        逻辑说明:
            1. 输入特征通过两个1x1卷积分为主分支和残差分支
            2. 主分支经过num_blocks个瓶颈模块提取特征
            3. 主分支与残差分支相加融合
            4. 通过1x1卷积调整通道数至out_channels
        """
        """
        前向传播（split拆分方式）
        Args:
            x: 输入特征图 [B, c1, H, W]
        Returns:
            torch.Tensor: 融合后的特征图 [B, c2, H, W]
        逻辑说明:
            1. 通过1x1卷积将输入通道扩展至c3
            2. 按通道均分为两个分支（split方式）
            3. 分支1经过CSP层+3x3卷积得到特征A
            4. 分支2经过CSP层+3x3卷积得到特征B
            5. 合并原始分支、特征A、特征B后通过1x1卷积输出
        """
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """
    类VGG网络块（融合3x3和1x1卷积）
    通过多尺度卷积核融合提升特征表达能力，支持部署时结构优化
    Args:
        ch_in (int): 输入通道数
        ch_out (int): 输出通道数
    """
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: 输入特征张量 [B, in_channels, H, W]
        Returns:
            torch.Tensor: 输出特征张量 [B, out_channels, H, W]
        逻辑说明:
            1. 输入特征通过两个1x1卷积分为主分支和残差分支
            2. 主分支经过num_blocks个瓶颈模块提取特征
            3. 主分支与残差分支相加融合
            4. 通过1x1卷积调整通道数至out_channels
        """
        """
        前向传播（split拆分方式）
        Args:
            x: 输入特征图 [B, c1, H, W]
        Returns:
            torch.Tensor: 融合后的特征图 [B, c2, H, W]
        逻辑说明:
            1. 通过1x1卷积将输入通道扩展至c3
            2. 按通道均分为两个分支（split方式）
            3. 分支1经过CSP层+3x3卷积得到特征A
            4. 分支2经过CSP层+3x3卷积得到特征B
            5. 合并原始分支、特征A、特征B后通过1x1卷积输出
        """
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    """
    CSP（Cross Stage Partial）特征层
    通过通道拆分与部分网络处理提升特征复用效率
    支持自定义瓶颈模块（默认使用VGGBlock）
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        num_blocks (int): 瓶颈模块重复次数
        expansion (float): 中间通道扩展系数
        bias (bool): 卷积层是否使用偏置
        act (str): 激活函数类型
        bottletype (nn.Module): 瓶颈模块类型（如VGGBlock）
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        """
        初始化CSP层
        Args:
            in_channels: 输入特征的通道数
            out_channels: 输出特征的通道数
            num_blocks: 瓶颈模块的堆叠次数（控制特征提取深度）
            expansion: 中间隐藏通道的扩展比例（隐藏通道=out_channels*expansion）
            bias: 卷积层是否添加偏置项（通常与归一化层配合使用时设为False）
            act: 激活函数类型（如"silu"增强非线性表达）
            bottletype: 瓶颈模块类（决定特征变换方式，默认使用VGGBlock）
        """
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: 输入特征张量 [B, in_channels, H, W]
        Returns:
            torch.Tensor: 输出特征张量 [B, out_channels, H, W]
        逻辑说明:
            1. 输入特征通过两个1x1卷积分为主分支和残差分支
            2. 主分支经过num_blocks个瓶颈模块提取特征
            3. 主分支与残差分支相加融合
            4. 通过1x1卷积调整通道数至out_channels
        """
        """
        前向传播（split拆分方式）
        Args:
            x: 输入特征图 [B, c1, H, W]
        Returns:
            torch.Tensor: 融合后的特征图 [B, c2, H, W]
        逻辑说明:
            1. 通过1x1卷积将输入通道扩展至c3
            2. 按通道均分为两个分支（split方式）
            3. 分支1经过CSP层+3x3卷积得到特征A
            4. 分支2经过CSP层+3x3卷积得到特征B
            5. 合并原始分支、特征A、特征B后通过1x1卷积输出
        """
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    """
    改进型CSP-ELAN特征融合模块（RepNCSPELAN4）
    通过多分支特征提取与跨阶段融合提升特征表达能力
    支持chunk和split两种特征拆分方式（对应不同部署需求）
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        c3 (int): 中间过渡通道数
        c4 (int): 分支卷积通道数
        n (int): 分支模块重复次数
        bias (bool): 卷积层是否使用偏置
        act (str): 激活函数类型
    """
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        """
        初始化RepNCSPELAN4模块
        Args:
            c1: 输入特征图的通道数
            c2: 输出特征图的通道数
            c3: 1x1卷积过渡的中间通道数（拆分为两个分支）
            c4: 分支CSP层的输出通道数
            n: CSP层内部块的重复次数（控制网络深度）
            bias: 所有卷积层是否启用偏置（部署优化时通常设为False）
            act: 激活函数类型（如"silu"或"relu"）
        """
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: 输入特征张量 [B, in_channels, H, W]
        Returns:
            torch.Tensor: 输出特征张量 [B, out_channels, H, W]
        逻辑说明:
            1. 输入特征通过两个1x1卷积分为主分支和残差分支
            2. 主分支经过num_blocks个瓶颈模块提取特征
            3. 主分支与残差分支相加融合
            4. 通过1x1卷积调整通道数至out_channels
        """
        """
        前向传播（split拆分方式）
        Args:
            x: 输入特征图 [B, c1, H, W]
        Returns:
            torch.Tensor: 融合后的特征图 [B, c2, H, W]
        逻辑说明:
            1. 通过1x1卷积将输入通道扩展至c3
            2. 按通道均分为两个分支（split方式）
            3. 分支1经过CSP层+3x3卷积得到特征A
            4. 分支2经过CSP层+3x3卷积得到特征B
            5. 合并原始分支、特征A、特征B后通过1x1卷积输出
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# transformer
class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层，实现自注意力机制与前馈网络的核心计算模块
    支持预归一化（pre-normalization）和后归一化（post-normalization）两种模式
    参考论文《On Layer Normalization in the Transformer Architecture》设计
    Args:
        d_model (int): 输入特征维度（模型维度）
        nhead (int): 多头注意力的头数
        dim_feedforward (int): 前馈网络的中间层维度
        dropout (float): 丢弃层概率
        activation (str): 激活函数类型
        normalize_before (bool): 是否使用预归一化模式
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        """
        初始化Transformer编码器层
        Args:
            d_model: 输入/输出特征的维度大小（与多头注意力的维度一致）
            nhead: 注意力头数，d_model需能被nhead整除
            dim_feedforward: 前馈网络中间层的维度（通常为d_model的4倍）
            dropout: 自注意力和前馈网络中的丢弃概率
            activation: 前馈网络的激活函数（如"relu"或"gelu"）
            normalize_before: 归一化位置标志位（True为预归一化，False为后归一化）
        """
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        """
        前向传播计算
        Args:
            src: 输入特征张量 [B, N, d_model]（B-批次，N-序列长度）
            src_mask: 注意力掩码（用于屏蔽无效位置）
            pos_embed: 位置编码张量 [1, N, d_model]
        Returns:
            torch.Tensor: 编码后的特征张量 [B, N, d_model]
        逻辑说明:
            1. 根据normalize_before标志选择归一化顺序
            2. 融合位置编码后计算自注意力
            3. 残差连接+丢弃层后进行前馈网络计算
            4. 最终输出经过归一化的编码特征
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        """
        前向传播计算
        Args:
            src: 输入特征张量 [B, N, d_model]（B-批次，N-序列长度）
            src_mask: 注意力掩码（用于屏蔽无效位置）
            pos_embed: 位置编码张量 [1, N, d_model]
        Returns:
            torch.Tensor: 编码后的特征张量 [B, N, d_model]
        逻辑说明:
            1. 根据normalize_before标志选择归一化顺序
            2. 融合位置编码后计算自注意力
            3. 残差连接+丢弃层后进行前馈网络计算
            4. 最终输出经过归一化的编码特征
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    """
    混合编码器，结合卷积网络和Transformer编码器的多尺度特征提取模块
    主要用于目标检测任务中，融合不同层级的特征图并通过Transformer增强上下文建模
    Args:
        in_channels (list): 输入特征图的通道数列表（对应不同层级）
        feat_strides (list): 输入特征图的步长列表（对应感受野大小）
        hidden_dim (int): 隐藏层维度，统一特征图通道数
        nhead (int): Transformer编码器的注意力头数
        dim_feedforward (int): Transformer前馈网络的中间维度
        dropout (float):  dropout概率
        enc_act (str): Transformer编码器激活函数类型
        use_encoder_idx (list): 需要应用Transformer编码器的特征层级索引
        num_encoder_layers (int): 每个Transformer编码器的层数
        pe_temperature (int): 位置编码的温度参数（控制频率）
        expansion (float): CSP层的通道扩展系数
        depth_mult (float): 网络深度乘数（控制模块重复次数）
        act (str): 卷积模块的激活函数类型
        eval_spatial_size (tuple): 评估时固定输入尺寸（用于预生成位置编码）
        version (str): 版本标识（控制FPN/PAN模块类型）
    """
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],  # 对应不同层级的特征图，P3、P4、P5
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2], # 决定哪个层用 encoder，也就是用 transformer 增强上下文信息
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 ):
        """
        初始化混合编码器
        主要完成输入投影层、Transformer编码器、FPN/PAN特征融合模块的初始化
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # 输入投影层，将不同层级的特征图映射到统一维度
        # 这里使用了 nn.ModuleList 来存储每个层级的投影层，
        # 每个投影层包含一个卷积层和一个批量归一化层，用于将输入特征图映射到隐藏层维度
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))

            self.input_proj.append(proj)

        # Transformer 编码器，类似于 RT-DETR 中的 AIFI 模块，对指定的层进行增强，deim 只对 32 倍下采样的特征图进行增强
        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down FPN（特征金字塔网络）：自上而下的特征融合模块
        # 功能：将高层语义强但分辨率低的特征与低层细节丰富但语义弱的特征融合，提升多尺度目标检测能力
        # 关键参数：
        #   lateral_convs: 1x1卷积层列表，用于调整高层特征通道数，保持与低层特征通道一致
        #   fpn_blocks: 特征融合块列表（如RepNCSPELAN4或CSPLayer），拼接上采样特征与低层特征后进一步融合
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        # 从最高层（索引最大）向低层（索引0）遍历，构建FPN路径
        for _ in range(len(in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            # 根据版本选择是否添加激活函数：dfine版本默认无激活，其他版本使用指定激活函数
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            # 特征融合块：输入为上采样特征（来自高层）与原始低层特征的拼接（通道数2*hidden_dim）
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up PAN（路径聚合网络）：自下而上的特征增强模块
        # 功能：在FPN基础上补充自下而上的特征传播路径，增强低层特征的上下文信息，提升小目标检测能力
        # 关键参数：
        #   downsample_convs: 下采样卷积层列表（如SCDown或3x3步长2卷积），用于将低层特征下采样至高层分辨率
        #   pan_blocks: 特征增强块列表（同FPN的融合块），拼接下采样特征与FPN输出的高层特征后进一步融合
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        # 从低层（索引0）向高层（索引最大-1）遍历，构建PAN路径
        for _ in range(len(in_channels) - 1):
            # 下采样卷积：dfine版本使用SCDown（空间压缩下采样），其他版本使用标准3x3步长2卷积
            # self.downsample_convs.append(
            #     nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
            #     if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            # )
            self.downsample_convs.append(
                nn.Sequential(ContextGuidedBlock_Down(hidden_dim), ConvNormLayer_fuse(hidden_dim* 2, hidden_dim, 1, 1)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            # 特征增强块：输入为下采样特征（来自低层）与FPN输出的高层特征的拼接（通道数2*hidden_dim）
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        构建2D正弦余弦位置编码（经典ViT位置编码方法）
        数学公式：
            pos_dim = embed_dim // 4
            omega = 1 / (temperature^(k/pos_dim)), k=0,1,...,pos_dim-1
            pos_emb(x,y) = [sin(x*omega), cos(x*omega), sin(y*omega), cos(y*omega)]
        Args:
            w (int): 特征图宽度（像素数）
            h (int): 特征图高度（像素数）
            embed_dim (int): 位置编码维度
            temperature (int): 温度参数（控制高频分量密度）
        Returns:
            torch.Tensor: 位置编码张量 [1, w*h, embed_dim]
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        """
        前向传播计算（混合编码器核心逻辑）
        Args:
            feats (list[torch.Tensor]): 输入特征图列表，每个元素形状为[B, in_channels[i], H_i, W_i]
                （B-批次大小，in_channels[i]-第i层输入通道数，H_i/W_i-第i层特征图高/宽）
        Returns:
            list[torch.Tensor]: 输出融合特征图列表，每个元素形状为[B, hidden_dim, H_out, W_out]
                （hidden_dim-统一隐藏维度，H_out/W_out-对应层级输出高/宽）
        逻辑说明:
            1. 输入投影：将不同层级特征图投影到统一隐藏维度hidden_dim
            2. Transformer编码：对指定层级特征应用Transformer增强上下文信息
            3. FPN特征融合：自上而下融合高层语义特征与低层细节特征
            4. PAN特征增强：自下而上补充低层特征的上下文信息
        """
        assert len(feats) == len(self.in_channels), "输入特征层数与配置不匹配"
        # 输入投影：通过1x1卷积+BN将不同层级特征图通道统一为hidden_dim
        # 输入形状：[B, in_channels[i], H_i, W_i] → 输出形状：[B, hidden_dim, H_i, W_i]
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # -------------------------- Transformer编码阶段 --------------------------
        # 对指定层级特征应用Transformer编码器，增强长距离依赖建模能力
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                # 获取当前层级特征图尺寸（H, W）
                h, w = proj_feats[enc_ind].shape[2:]
                # 特征展平：[B, hidden_dim, H, W] → [B, H*W, hidden_dim]（序列长度=H*W）
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                # 位置编码生成：训练时动态生成，评估时使用预计算编码（提升推理速度）
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                # Transformer编码：[B, H*W, hidden_dim] → [B, H*W, hidden_dim]
                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                # 恢复特征图形状：[B, H*W, hidden_dim] → [B, hidden_dim, H, W]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # -------------------------- FPN特征融合阶段 --------------------------
        # 自上而下特征金字塔网络（Feature Pyramid Network）
        # 功能：将高层语义强但分辨率低的特征与低层细节丰富的特征融合
        # 初始化inner_outs为最高层特征（如P5，32倍下采样），形状[B, hidden_dim, H5, W5]
        inner_outs = [proj_feats[-1]]
        # 从最高层（索引len-1）向低层（索引1）遍历，处理len-1次融合（如P5→P4→P3）
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]  # 当前最高层特征（如P5处理后结果）
            feat_low = proj_feats[idx - 1]  # 待融合的低层原始特征（如P4原始特征）
            # 1x1卷积调整通道：保持与低层特征通道一致（hidden_dim）
            # 输入形状：[B, hidden_dim, H_heigh, W_heigh] → 输出形状不变
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh  # 更新最高层特征为调整后结果
            # 最近邻上采样：尺寸放大2倍（H_heigh*2, W_heigh*2），与低层特征尺寸匹配
            # 输入形状：[B, hidden_dim, H_heigh, W_heigh] → 输出形状：[B, hidden_dim, 2H_heigh, 2W_heigh]
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # 特征拼接与融合：拼接上采样特征（高层）与原始低层特征（通道数2*hidden_dim）
            # 通过RepNCSPELAN4/CSPLayer进一步融合，输出通道恢复为hidden_dim
            # 输入形状：[B, 2*hidden_dim, H_low, W_low] → 输出形状：[B, hidden_dim, H_low, W_low]
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)  # 插入列表头部，对应更低层特征（如P4处理结果）

        # -------------------------- PAN特征增强阶段 --------------------------
        # 自下而上路径聚合网络（Path Aggregation Network）
        # 功能：补充自下而上的特征传播路径，增强低层特征的上下文信息
        # 初始化outs为FPN最底层输出（如P3处理结果），形状[B, hidden_dim, H3, W3]
        outs = [inner_outs[0]]
        # 从底层（索引0）向高层（索引len-2）遍历，处理len-1次增强（如P3→P4→P5）
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]  # 当前最底层特征（如P3增强结果）
            feat_height = inner_outs[idx + 1]  # FPN对应高层特征（如P4处理结果）
            # 下采样：通过SCDown或3x3步长2卷积缩小尺寸（H_low/2, W_low/2）
            # 输入形状：[B, hidden_dim, H_low, W_low] → 输出形状：[B, hidden_dim, H_low/2, W_low/2]
            downsample_feat = self.downsample_convs[idx](feat_low)
            # 特征拼接与增强：拼接下采样特征（低层）与FPN高层特征（通道数2*hidden_dim）
            # 通过RepNCSPELAN4/CSPLayer进一步融合，输出通道保持hidden_dim
            # 输入形状：[B, 2*hidden_dim, H_height, W_height] → 输出形状：[B, hidden_dim, H_height, W_height]
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)  # 添加到列表尾部，对应更高层增强特征（如P4增强结果）

        return outs

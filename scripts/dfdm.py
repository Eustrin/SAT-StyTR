import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 引入上一级 StyTR-2 模块

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAttentionBranch(nn.Module):
    """对应论文 1.2 节 & 图 2(a): 3级级联最大池化(3x3->2x2) + 13x13/3x3交替卷积 + 双线性重加权"""
    def __init__(self, in_ch=512, mid_ch=64):
        super().__init__()
        # 3级最大池化下采样: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 1x1降维防参数爆炸 + 交替堆叠 13x13 大核与 3x3 小核
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),  # 空间归一化为像素级注意力权重图
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        feat = self.pool(x)
        attn = self.conv_net(feat)
        attn = F.interpolate(attn, size=(h, w), mode="bilinear", align_corners=False)
        return x * attn, attn


class StyleAttentionBranch(nn.Module):
    """对应论文 1.2 节 & 2.4.2 节 & 图 2(b): 5级渐进平均池化(5x5->3x3) + 13x13/3x3交替卷积"""
    def __init__(self, in_ch=512, mid_ch=64):
        super().__init__()
        # 5级平均池化: 第1/3/5级 stride=2 降采样(32->16->8->4), 第2/4级 stride=1 平滑滤波
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        feat = self.pool(x)
        attn = self.conv_net(feat)
        attn = F.interpolate(attn, size=(h, w), mode="bilinear", align_corners=False)
        return x * attn, attn


class DFDM(nn.Module):
    """单个双模态特征解耦模块 (图 1 里重复 3 次: DFDM×3)"""
    def __init__(self, channels=512, mid_ch=64):
        super().__init__()
        self.content_branch = ContentAttentionBranch(channels, mid_ch)
        self.style_branch = StyleAttentionBranch(channels, mid_ch)

    def forward(self, x_c, x_s):
        out_c, attn_c = self.content_branch(x_c)
        out_s, attn_s = self.style_branch(x_s)
        return out_c, out_s


if __name__ == "__main__":
    from dataset_cuhk import PairedCUHKDataset
    from models.StyTR import PatchEmbed

    # 1. 取真实训练数据 1 对
    ds = PairedCUHKDataset(size=256)
    ic, is_ = ds[0]
    ic = ic.unsqueeze(0)   # [1, 3, 256, 256]
    is_ = is_.unsqueeze(0) # [1, 3, 256, 256]

    # 2. 过原版 StyTR-2 的分块嵌入 (PatchEmbed)
    embed = PatchEmbed(img_size=256, patch_size=8, in_chans=3, embed_dim=512)
    feat_c = embed(ic)
    feat_s = embed(is_)
    print("PatchEmbed 输出形状:", feat_c.shape)

    # 3. 过图 1 里的 DFDM×3
    dfdm_blocks = nn.ModuleList([DFDM(512, mid_ch=64) for _ in range(3)])
    params = sum(p.numel() for p in dfdm_blocks.parameters()) / 1e6
    print(f"DFDM×3 总参数量: {params:.2f} M")

    for block in dfdm_blocks:
        feat_c, feat_s = block(feat_c, feat_s)

    print("DFDM×3 输出内容特征:", feat_c.shape, "风格特征:", feat_s.shape)

    # 4. 测试反向传播梯度是否正常
    loss = feat_c.mean() + feat_s.mean()
    loss.backward()
    print("DFDM 梯度反传成功!")

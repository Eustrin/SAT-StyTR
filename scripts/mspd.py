import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


def _make_norm(norm_type: str, num_features: int):
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm2d(num_features)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class SubDiscriminator(nn.Module):
    """对应论文 1.4 节 & 图 3: 单尺度子判别器，输出 Sigmoid 判别图并保留中间层特征"""
    def __init__(self, in_ch=3, base_ch=64, norm_type="instance", use_sigmoid=True):
        super().__init__()
        # 工程折中：论文未给出子判别器具体层数/通道数，采用 4 层 PatchGAN 风格卷积 (64->128->256->512->1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
                _make_norm(norm_type, base_ch * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
                _make_norm(norm_type, base_ch * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=1, padding=1),
                _make_norm(norm_type, base_ch * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        self.head = nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1)
        self.gate = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x):
        feats = []
        h = x
        for blk in self.blocks:
            h = blk(h)
            feats.append(h)
        prob = self.gate(self.head(h))
        return prob, feats


class MSPD(nn.Module):
    """对应论文 1.4 节 & 图 3 & 公式(14): 3 尺度多尺度感知判别器"""
    def __init__(self, in_ch=3, base_ch=64, num_scales=3, norm_type="instance", use_sigmoid=True):
        super().__init__()
        assert num_scales == 3, "论文公式(14) 固定为 3 个尺度判别器"
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.discriminators = nn.ModuleList([
            SubDiscriminator(in_ch=in_ch, base_ch=base_ch, norm_type=norm_type, use_sigmoid=use_sigmoid)
            for _ in range(num_scales)
        ])

    def forward(self, x):
        probs = []
        all_feats = []
        cur = x
        for idx, disc in enumerate(self.discriminators):
            if idx > 0:
                cur = self.downsample(cur)  # 256 -> 128 -> 64
            prob, feats = disc(cur)
            probs.append(prob)
            all_feats.append(feats)
        return probs, all_feats


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    mspd = MSPD(in_ch=3, base_ch=64, num_scales=3, norm_type="instance", use_sigmoid=True)
    params = sum(p.numel() for p in mspd.parameters()) / 1e6
    print(f"MSPD 总参数量: {params:.2f} M")

    probs, all_feats = mspd(x)
    for i, (p, fs) in enumerate(zip(probs, all_feats), 1):
        print(f"尺度 {i}: D_{i} 输出={tuple(p.shape)} 范围=({p.min().item():.4f}, {p.max().item():.4f}) 中间特征层数={len(fs)}")

    loss = sum(-torch.log(torch.clamp(p, 1e-6, 1.0 - 1e-6)).mean() for p in probs) / len(probs)
    loss.backward()
    print("MSPD 梯度反传成功!")

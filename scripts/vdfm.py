import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


class VDFM(nn.Module):
    """对应论文 1.3 节公式(1)~(8)，图1 中重复 3 次：VDFM×3"""
    def __init__(self, channels=512, num_heads=8, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 公式(1): Concat(X1, X2) 后 2C 维过 SA，再 Linear 回 C
        self.sa = nn.MultiheadAttention(embed_dim=channels * 2, num_heads=num_heads, batch_first=True)
        self.linear_f = nn.Linear(channels * 2, channels)
        # 公式(2): X2 线性投影
        self.linear_s = nn.Linear(channels, channels)

    def axis_norm(self, x, dim):
        # 公式(3): L1 / L2 / L_inf 多范数分支；工程折中：三分支取均值
        N = x.shape[dim]
        l1 = x.abs().sum(dim=dim, keepdim=True) / N
        l2 = torch.sqrt((x ** 2).sum(dim=dim, keepdim=True) + self.eps) / N
        linf = x.abs().amax(dim=dim, keepdim=True)
        return (l1 + l2 + linf) / 3.0

    def gate_fuse(self, xf, x2_p, dim):
        # 公式(4)(5) 指数增强型归一化门控（clamp 防 exp 溢出）；公式(6)(7) 加权融合
        a_f = torch.clamp(self.axis_norm(xf, dim), max=20.0)
        a_s = torch.clamp(self.axis_norm(x2_p, dim), max=20.0)
        exp_f, exp_s = torch.exp(a_f), torch.exp(a_s)
        denom = exp_f + exp_s + self.eps
        return (exp_f / denom) * xf + (exp_s / denom) * x2_p

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        cat_seq = torch.cat([x1, x2], dim=1).flatten(2).transpose(1, 2)      # [B, HW, 2C]
        sa_out, _ = self.sa(cat_seq, cat_seq, cat_seq)
        xf = self.linear_f(sa_out).transpose(1, 2).view(B, C, H, W)          # X_f
        x2_p = self.linear_s(x2.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        f1 = self.gate_fuse(xf, x2_p, dim=-1)                                # 行向量融合 F1
        f2 = self.gate_fuse(xf, x2_p, dim=-2)                                # 列向量融合 F2
        return f1 + f2                                                       # 公式(8)


if __name__ == "__main__":
    from dataset_cuhk import PairedCUHKDataset
    from dfdm import DFDM
    from models.StyTR import PatchEmbed, decoder

    ds = PairedCUHKDataset(size=256)
    ic, is_ = ds[0]
    ic, is_ = ic.unsqueeze(0), is_.unsqueeze(0)

    embed = PatchEmbed(img_size=256, patch_size=8, in_chans=3, embed_dim=512)
    dfdm_blocks = nn.ModuleList([DFDM(512, mid_ch=64) for _ in range(3)])
    vdfm_blocks = nn.ModuleList([VDFM(512, num_heads=8) for _ in range(3)])
    print(f"VDFM×3 总参数量: {sum(p.numel() for p in vdfm_blocks.parameters()) / 1e6:.2f} M")

    feat_c, feat_s = embed(ic), embed(is_)
    for blk in dfdm_blocks:
        feat_c, feat_s = blk(feat_c, feat_s)

    fused = feat_c
    for blk in vdfm_blocks:
        fused = blk(fused, feat_s)

    gs = decoder(fused)
    print("生成图像 Gs 形状:", gs.shape)
    gs.mean().backward()
    print("整条生成器链路 (PatchEmbed -> DFDM×3 -> VDFM×3 -> Decoder) 梯度反传成功!")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from function import calc_mean_std, normal
import models.StyTR as StyTR

BASE_DIR = Path(__file__).resolve().parent.parent


def load_frozen_vgg_stages(vgg_path=None):
    """复用原版 models/StyTR.py 的预训练 VGG19 前 5 阶段提取器 Φ_1..Φ_5（冻结参数）"""
    vgg_path = Path(vgg_path) if vgg_path else (BASE_DIR / "experiments" / "vgg_normalised.pth")
    vgg = StyTR.vgg
    if vgg_path.exists():
        vgg.load_state_dict(torch.load(vgg_path, map_location="cpu", weights_only=False))
    vgg_44 = nn.Sequential(*list(vgg.children())[:44])
    enc_layers = list(vgg_44.children())
    stages = nn.ModuleList([
        nn.Sequential(*enc_layers[:4]),    # relu1_1
        nn.Sequential(*enc_layers[4:11]),  # relu2_1
        nn.Sequential(*enc_layers[11:18]), # relu3_1
        nn.Sequential(*enc_layers[18:31]), # relu4_1
        nn.Sequential(*enc_layers[31:44]), # relu5_1
    ])
    for p in stages.parameters():
        p.requires_grad = False
    stages.eval()
    return stages


def _make_gaussian_kernel(kernel_size=5, sigma=1.0):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d = torch.outer(g, g)
    g2d = g2d / g2d.sum()
    return g2d.view(1, 1, kernel_size, kernel_size)


def _rgb_to_gray(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class LoGDetailLoss(nn.Module):
    """对应论文 1.5.3 节公式(11): L_D = ||m * f(lap(Gs)) - f(lap(Ic))||_2^2"""
    def __init__(self, edge_thresh_ratio=0.08):
        super().__init__()
        self.edge_thresh_ratio = edge_thresh_ratio
        lap = torch.tensor([[0.0, 1.0, 0.0],
                            [1.0, -4.0, 1.0],
                            [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        gauss = _make_gaussian_kernel(kernel_size=5, sigma=1.0)
        self.register_buffer("lap_kernel", lap)
        self.register_buffer("gauss_kernel", gauss)

    def _log_response(self, x):
        gray = _rgb_to_gray(x)
        lap_out = F.conv2d(gray, self.lap_kernel, padding=1)
        return F.conv2d(lap_out, self.gauss_kernel, padding=2)

    def forward(self, gs, ic):
        log_gs = self._log_response(gs)
        with torch.no_grad():
            log_ic = self._log_response(ic)
            # 工程折中：论文未写掩码 m 的具体生成规则，这里对 |f(lap(Ic))| 按峰值比例阈值化得到二值边缘掩码
            mag = log_ic.abs()
            peak = mag.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            m = (mag >= self.edge_thresh_ratio * peak).float()
        return F.mse_loss(m * log_gs, log_ic)


class SATLoss(nn.Module):
    """对应论文 1.5 节公式(9)~(15): 6 项加权总损失 + 判别器损失"""
    def __init__(
        self,
        vgg_path=None,
        lambda_c=7.0,
        lambda_s=18.0,
        lambda_d=200.0,
        lambda_id1=70.0,
        lambda_id2=1.0,
        lambda_a=1.0,
        gray_ic_for_id1=True,
    ):
        super().__init__()
        self.vgg_stages = load_frozen_vgg_stages(vgg_path)
        self.log_loss = LoGDetailLoss()
        self.mse = nn.MSELoss()

        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_d = lambda_d
        self.lambda_id1 = lambda_id1
        self.lambda_id2 = lambda_id2
        self.lambda_a = lambda_a
        # 工程折中（待向导师确认疑点②）：Ic 转 3 通道灰度再与素描 Gs 算像素 MSE，避免强迫素描输出彩色肤色
        self.gray_ic_for_id1 = gray_ic_for_id1

    def extract_vgg(self, x):
        feats = []
        h = x
        for stage in self.vgg_stages:
            h = stage(h)
            feats.append(h)
        return feats

    def calc_content_loss(self, gs_feats, ic_feats):
        # 公式(9) + 原版 StyTR.py:209：VGG 第 4、5 层特征归一化后算 MSE
        return (
            self.mse(normal(gs_feats[-1]), normal(ic_feats[-1]))
            + self.mse(normal(gs_feats[-2]), normal(ic_feats[-2]))
        )

    def calc_style_loss(self, gs_feats, is_feats):
        # 公式(10) + 原版 StyTR.py:211-213：5 层通道均值与标准差 MSE
        loss = 0.0
        for gf, sf in zip(gs_feats, is_feats):
            g_mean, g_std = calc_mean_std(gf)
            s_mean, s_std = calc_mean_std(sf)
            loss = loss + self.mse(g_mean, s_mean) + self.mse(g_std, s_std)
        return loss

    def calc_id1_loss(self, gs, ic, is_):
        # 公式(12)
        ic_ref = _rgb_to_gray(ic).repeat(1, 3, 1, 1) if self.gray_ic_for_id1 else ic
        return self.mse(gs, ic_ref) + self.mse(gs, is_)

    def calc_id2_loss(self, gs_feats, ic_feats, is_feats):
        # 公式(13)
        loss = 0.0
        for gf, cf, sf in zip(gs_feats, ic_feats, is_feats):
            loss = loss + self.mse(gf, cf) + self.mse(gf, sf)
        return loss

    def calc_adv_loss(self, d_fake_probs):
        # 公式(14): (1/3N) * sum_i sum_j [-ln D_i(G(Ic))]
        if not d_fake_probs:
            return torch.tensor(0.0)
        return sum(-torch.log(torch.clamp(p, 1e-6, 1.0 - 1e-6)).mean() for p in d_fake_probs) / len(d_fake_probs)

    def calc_d_loss(self, d_real_probs, d_fake_probs):
        # MSPD 标准对抗判别器损失：真实素描趋近 1，生成素描趋近 0
        loss = 0.0
        for pr, pf in zip(d_real_probs, d_fake_probs):
            pr_c = torch.clamp(pr, 1e-6, 1.0 - 1e-6)
            pf_c = torch.clamp(pf, 1e-6, 1.0 - 1e-6)
            loss = loss + (-torch.log(pr_c).mean() - torch.log(1.0 - pf_c).mean()) * 0.5
        return loss / len(d_real_probs)

    def forward_generator(self, gs, ic, is_, d_fake_probs=None):
        gs_feats = self.extract_vgg(gs)
        with torch.no_grad():
            ic_feats = self.extract_vgg(ic)
            is_feats = self.extract_vgg(is_)

        l_c = self.calc_content_loss(gs_feats, ic_feats)
        l_s = self.calc_style_loss(gs_feats, is_feats)
        l_d = self.log_loss(gs, ic)
        l_id1 = self.calc_id1_loss(gs, ic, is_)
        l_id2 = self.calc_id2_loss(gs_feats, ic_feats, is_feats)
        l_a = self.calc_adv_loss(d_fake_probs) if d_fake_probs is not None else gs.new_zeros(())

        total = (
            self.lambda_c * l_c
            + self.lambda_s * l_s
            + self.lambda_d * l_d
            + self.lambda_id1 * l_id1
            + self.lambda_id2 * l_id2
            + self.lambda_a * l_a
        )
        details = {
            "L_total": total.item(),
            "L_C": l_c.item(),
            "L_S": l_s.item(),
            "L_D": l_d.item(),
            "L_ID1": l_id1.item(),
            "L_ID2": l_id2.item(),
            "L_A": l_a.item(),
        }
        return total, details


if __name__ == "__main__":
    from mspd import MSPD

    ic = torch.rand(1, 3, 256, 256)
    is_ = torch.rand(1, 3, 256, 256)
    gs = torch.rand(1, 3, 256, 256, requires_grad=True)

    mspd = MSPD()
    crit = SATLoss()
    fake_probs, _ = mspd(gs)
    real_probs, _ = mspd(is_)

    loss_g, details = crit.forward_generator(gs, ic, is_, fake_probs)
    loss_d = crit.calc_d_loss(real_probs, [p.detach() for p in fake_probs])
    print("生成器 6 项损失:", {k: round(v, 4) for k, v in details.items()})
    print("判别器损失 L_D_adv:", round(loss_d.item(), 4))

    loss_g.backward()
    assert gs.grad is not None
    print("SATLoss 梯度反传成功!")

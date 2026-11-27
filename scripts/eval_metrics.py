import argparse
import importlib.util
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent


def _gaussian_window(window_size=11, sigma=1.5, channels=3):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = (g / g.sum()).unsqueeze(1)
    g2d = g1d @ g1d.t()
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def calc_ssim(img1, img2, window_size=11):
    """纯 PyTorch 实现 SSIM（img1, img2 范围 [0, 1]，形状 [1, C, H, W]）"""
    C = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, C)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def match_gt_path(pred_path: Path, gt_dir: Path):
    stem = pred_path.stem
    if "_stylized_" in stem:
        stem = stem.split("_stylized_")[0]
    candidates = list(gt_dir.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="生成素描图像定量评估（论文 2.3 节表 1：SSIM/PSNR/MSE/LPIPS/FID）")
    parser.add_argument("--pred_dir", type=str, default=str(BASE_DIR / "out" / "sat"))
    parser.add_argument("--gt_dir", type=str, default=str(BASE_DIR / "data" / "cuhk" / "test" / "sketch"))
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pred_paths = sorted(p for p in pred_dir.glob("*.*") if p.is_file())
    assert pred_paths, f"未在 {pred_dir} 找到生成图像"

    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])

    has_lpips = importlib.util.find_spec("lpips") is not None
    lpips_fn = None
    if has_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex")
        lpips_fn.eval()

    ssims, psnrs, mses, lpips_vals = [], [], [], []
    for p_path in pred_paths:
        gt_path = match_gt_path(p_path, gt_dir)
        if gt_path is None:
            print(f"[跳过] 未找到真值匹配: {p_path.name}")
            continue

        pred = tf(Image.open(p_path).convert("RGB")).unsqueeze(0)
        gt = tf(Image.open(gt_path).convert("RGB")).unsqueeze(0)

        # 表 1 口径：MSE 在 [0, 255] 像素尺度计算（10*log10(255^2 / MSE) ≈ PSNR）
        mse_255 = F.mse_loss(pred * 255.0, gt * 255.0).item()
        psnr = 10.0 * math.log10((255.0 ** 2) / max(mse_255, 1e-8))
        ssim = calc_ssim(pred, gt)

        mses.append(mse_255)
        psnrs.append(psnr)
        ssims.append(ssim)

        if lpips_fn is not None:
            with torch.no_grad():
                val = lpips_fn(pred * 2.0 - 1.0, gt * 2.0 - 1.0).item()
            lpips_vals.append(val)

    assert ssims, f"未匹配到任何与 {gt_dir} 对应的生成图像"
    n = len(ssims)
    print(f"评估样本数: {n} (预测目录: {pred_dir})")
    print(f"SSIM ↑ : {sum(ssims) / n:.4f}")
    print(f"PSNR ↑ : {sum(psnrs) / n:.2f} dB")
    print(f"MSE  ↓ : {sum(mses) / n:.2f}")
    if lpips_vals:
        print(f"LPIPS ↓: {sum(lpips_vals) / n:.4f}")
    else:
        print("LPIPS ↓: N/A（未安装 lpips，需要时可执行 pip install lpips）")

    if importlib.util.find_spec("pytorch_fid") is not None:
        import numpy as np
        from scipy import linalg
        from pytorch_fid.fid_score import compute_statistics_of_path
        from pytorch_fid.inception import InceptionV3

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inc_model = InceptionV3([block_idx]).to("cpu")
        m1, s1 = compute_statistics_of_path(str(pred_dir), inc_model, min(n, 10), 2048, "cpu", 0)
        m2, s2 = compute_statistics_of_path(str(gt_dir), inc_model, min(n, 10), 2048, "cpu", 0)

        diff = m1 - m2
        covmean = linalg.sqrtm(s1.dot(s2))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
        if not np.isfinite(covmean).all():
            offset = np.eye(s1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
            if isinstance(covmean, tuple):
                covmean = covmean[0]
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
            covmean = covmean.real
        fid = float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
        print(f"FID  ↓ : {fid:.2f}")
    else:
        print("FID  ↓ : N/A（未安装 pytorch-fid，需要时可执行 pip install pytorch-fid）")


if __name__ == "__main__":
    main()

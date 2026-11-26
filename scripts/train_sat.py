import argparse
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from scripts.dataset_cuhk import PairedCUHKDataset
from scripts.mspd import MSPD
from scripts.sat_losses import SATLoss
from scripts.sat_model import SATStyTRGenerator

BASE_DIR = Path(__file__).resolve().parent.parent


def compute_lr_factor(step, total_steps, warmup_ratio=0.2, min_lr_scale=0.1):
    """
    对应论文 2.1 节学习率调度:
    前 20% 迭代（论文 <10^4 / 5e4）线性预热；后 80% 迭代指数衰减。
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return 0.1 + 0.9 * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    gamma = min_lr_scale ** (1.0 / decay_steps)
    return max(min_lr_scale, gamma ** (step - warmup_steps + 1))


def set_optimizer_lr(optimizer, base_lr, factor):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * factor


def main():
    parser = argparse.ArgumentParser(description="SAT-StyTR 轻量化 CPU 训练脚本（论文 2.1 节）")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR / "data" / "cuhk" / "train"))
    parser.add_argument("--steps", type=int, default=500, help="总训练步数（CPU 轻量实验建议 200~500）")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr_g", type=float, default=8e-4, help="生成器初始学习率（论文 2.1 节）")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="判别器初始学习率（论文 2.1 节）")
    parser.add_argument("--save_dir", type=str, default=str(BASE_DIR / "out" / "checkpoints"))
    parser.add_argument("--preview_dir", type=str, default=str(BASE_DIR / "out" / "train_preview"))
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--no_pretrained", action="store_true", help="不加载原版 StyTR-2 预训练权重")
    parser.add_argument("--no_dfdm", action="store_true", help="消融开关：关闭 DFDM")
    parser.add_argument("--no_vdfm", action="store_true", help="消融开关：关闭 VDFM")
    parser.add_argument("--no_mspd", action="store_true", help="消融开关：关闭 MSPD")
    parser.add_argument("--no_tr_enc", action="store_true", help="消融开关：关闭 Transformer 双路编码器")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    preview_dir = Path(args.preview_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    ds = PairedCUHKDataset(root=args.data_dir, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model_cfg = {
        "img_size": args.size,
        "use_dfdm": not args.no_dfdm,
        "use_vdfm": not args.no_vdfm,
        "use_transformer_encoder": not args.no_tr_enc,
    }
    gen = SATStyTRGenerator(**model_cfg)
    if not args.no_pretrained:
        gen.load_pretrained_stytr(BASE_DIR / "experiments")

    use_mspd = not args.no_mspd
    disc = MSPD() if use_mspd else None
    crit = SATLoss()

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999)) if use_mspd else None

    print(f"样本数={len(ds)}, batch_size={args.batch_size}, steps={args.steps}, use_mspd={use_mspd}")
    print(f"G 参数量: {sum(p.numel() for p in gen.parameters()) / 1e6:.2f} M"
          + (f", D 参数量: {sum(p.numel() for p in disc.parameters()) / 1e6:.2f} M" if use_mspd else ""))

    data_iter = iter(dl)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        try:
            ic, is_ = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            ic, is_ = next(data_iter)

        lr_factor = compute_lr_factor(step - 1, args.steps)
        set_optimizer_lr(opt_g, args.lr_g, lr_factor)
        if opt_d is not None:
            set_optimizer_lr(opt_d, args.lr_d, lr_factor)

        gen.train()
        if disc is not None:
            disc.train()

        # 1. 前向生成
        gs = gen(ic, is_)

        # 2. 更新判别器 MSPD
        loss_d_val = 0.0
        if use_mspd:
            opt_d.zero_grad()
            real_probs, _ = disc(is_)
            fake_probs_det, _ = disc(gs.detach())
            loss_d = crit.calc_d_loss(real_probs, fake_probs_det)
            loss_d.backward()
            opt_d.step()
            loss_d_val = loss_d.item()

        # 3. 更新生成器（6 项加权损失公式 15）
        opt_g.zero_grad()
        fake_probs = disc(gs)[0] if use_mspd else None
        loss_g, details = crit.forward_generator(gs, ic, is_, fake_probs)
        loss_g.backward()
        opt_g.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.time() - t0
            print(
                f"[{step:4d}/{args.steps}] "
                f"L_G={details['L_total']:.3f} (C={details['L_C']:.2f}, S={details['L_S']:.3f}, "
                f"D={details['L_D']:.4f}, ID1={details['L_ID1']:.3f}, ID2={details['L_ID2']:.2f}, A={details['L_A']:.3f}) "
                f"L_D={loss_d_val:.3f} lr_x={lr_factor:.2f} ({elapsed:.1f}s)"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "model_cfg": model_cfg,
                "generator": gen.state_dict(),
                "discriminator": disc.state_dict() if disc is not None else None,
            }
            torch.save(ckpt, save_dir / "sat_latest.pth")
            torch.save(ckpt, save_dir / f"sat_step_{step}.pth")
            with torch.no_grad():
                vis = torch.cat([ic[:1], is_[:1], torch.clamp(gs[:1], 0.0, 1.0)], dim=3)
                save_image(vis, preview_dir / f"preview_step_{step:04d}.png")

    print(f"训练完成！最新权重: {save_dir / 'sat_latest.pth'}")


if __name__ == "__main__":
    main()

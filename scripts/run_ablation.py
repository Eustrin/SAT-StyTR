import argparse
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ABLATION_VARIANTS = [
    ("no_dfdm", "w/o DFDM", ["--no_dfdm"]),
    ("no_vdfm", "w/o VDFM", ["--no_vdfm"]),
    ("no_dfdm_vdfm", "w/o DFDM+VDFM", ["--no_dfdm", "--no_vdfm"]),
    ("no_mspd", "w/o MSPD", ["--no_mspd"]),
    ("no_tr_enc", "w/o Transformer 编码器", ["--no_tr_enc"]),
]


def parse_metrics(output: str):
    ssim = re.search(r"SSIM\s*↑\s*:\s*([0-9.]+)", output)
    psnr = re.search(r"PSNR\s*↑\s*:\s*([0-9.]+)", output)
    mse = re.search(r"MSE\s*↓\s*:\s*([0-9.]+)", output)
    lpips_m = re.search(r"LPIPS\s*↓:\s*([0-9.]+)", output)
    fid_m = re.search(r"FID\s*↓\s*:\s*([0-9.]+)", output)
    return (
        ssim.group(1) if ssim else "N/A",
        psnr.group(1) if psnr else "N/A",
        mse.group(1) if mse else "N/A",
        lpips_m.group(1) if lpips_m else "N/A",
        fid_m.group(1) if fid_m else "N/A",
    )


def main():
    parser = argparse.ArgumentParser(description="批量运行并评估 SAT-StyTR 5 组消融实验")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--variants", nargs="*", default=[v[0] for v in ABLATION_VARIANTS])
    parser.add_argument("--force_train", action="store_true", help="即使已有权重也重新训练")
    args = parser.parse_args()

    selected = [v for v in ABLATION_VARIANTS if v[0] in set(args.variants)]
    assert selected, f"未匹配到有效消融变体: {args.variants}"

    summary = []
    for name, label, flags in selected:
        save_dir = BASE_DIR / "out" / "checkpoints_ablation" / name
        preview_dir = BASE_DIR / "out" / "train_preview_ablation" / name
        pred_dir = BASE_DIR / "out" / "sat_ablation" / name
        ckpt_path = save_dir / "sat_latest.pth"

        print(f"\n=== 消融实验: {label} ({name}) ===")
        if ckpt_path.exists() and not args.force_train:
            print(f"检测到已有权重，跳过训练: {ckpt_path}（如需重训请加 --force_train）")
        else:
            train_cmd = [
                sys.executable,
                str(BASE_DIR / "scripts" / "train_sat.py"),
                "--steps", str(args.steps),
                "--batch_size", str(args.batch_size),
                "--save_every", str(args.save_every),
                "--log_every", str(args.log_every),
                "--save_dir", str(save_dir),
                "--preview_dir", str(preview_dir),
                *flags,
            ]
            subprocess.run(train_cmd, check=True)

        infer_cmd = [
            sys.executable,
            str(BASE_DIR / "scripts" / "infer_sat.py"),
            "--ckpt", str(ckpt_path),
            "--output", str(pred_dir),
        ]
        subprocess.run(infer_cmd, check=True)

        eval_cmd = [
            sys.executable,
            str(BASE_DIR / "scripts" / "eval_metrics.py"),
            "--pred_dir", str(pred_dir),
        ]
        res = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
        print(res.stdout.strip())
        ssim, psnr, mse, lpips_v, fid_v = parse_metrics(res.stdout)
        summary.append((label, name, ssim, psnr, mse, lpips_v, fid_v))

    print("\n=== 消融实验定量汇总 ===")
    print(f"{'变体':<24} {'目录':<16} {'SSIM ↑':>8} {'PSNR/dB ↑':>10} {'MSE ↓':>10} {'LPIPS ↓':>9} {'FID ↓':>8}")
    for label, name, ssim, psnr, mse, lpips_v, fid_v in summary:
        print(f"{label:<24} {name:<16} {ssim:>8} {psnr:>10} {mse:>10} {lpips_v:>9} {fid_v:>8}")


if __name__ == "__main__":
    main()

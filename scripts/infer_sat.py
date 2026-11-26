import argparse
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from scripts.sat_model import SATStyTRGenerator

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="SAT-StyTR 推理脚本（论文 2.2 节图 4 & 表 5）")
    parser.add_argument("--ckpt", type=str, default=str(BASE_DIR / "out" / "checkpoints" / "sat_latest.pth"))
    parser.add_argument("--content_dir", type=str, default=str(BASE_DIR / "data" / "cuhk" / "test" / "photo"))
    parser.add_argument("--style", type=str, default=str(BASE_DIR / "data" / "cuhk" / "train" / "sketch" / "001_f-001-01.png"),
                        help="单张参考素描路径（默认取训练集第1张，禁止使用 test/sketch/）")
    parser.add_argument("--style_dir", type=str, default="", help="可选：批量风格目录（若指定则覆盖 --style）")
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "out" / "sat"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ext", type=str, default=".jpg")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"未找到训练权重: {ckpt_path}，请先运行 scripts/train_sat.py"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = ckpt.get("model_cfg", {"img_size": args.size})
    gen = SATStyTRGenerator(**model_cfg)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()

    content_paths = sorted(p for p in Path(args.content_dir).glob("*.*") if p.is_file())
    assert content_paths, f"未在 {args.content_dir} 找到内容图像"

    if args.style_dir:
        style_paths = sorted(p for p in Path(args.style_dir).glob("*.*") if p.is_file())
    else:
        style_paths = [Path(args.style)]
    assert style_paths and all(p.exists() for p in style_paths), "未找到有效风格参考图"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])

    total_time = 0.0
    count = 0
    with torch.no_grad():
        for c_path in content_paths:
            ic = tf(Image.open(c_path).convert("RGB")).unsqueeze(0)
            for s_path in style_paths:
                is_ = tf(Image.open(s_path).convert("RGB")).unsqueeze(0)
                t0 = time.perf_counter()
                gs = torch.clamp(gen(ic, is_), 0.0, 1.0)
                total_time += time.perf_counter() - t0
                count += 1
                out_name = f"{c_path.stem}_stylized_{s_path.stem}{args.ext}"
                save_image(gs, out_dir / out_name)

    print(f"推理完成！共生成 {count} 张图，输出目录: {out_dir}，平均耗时 t_inf={total_time / count:.3f} s/张")


if __name__ == "__main__":
    main()

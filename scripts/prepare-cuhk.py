import re
from pathlib import Path
from PIL import Image

SRC_PHOTO = Path("../cuhk-face-sketch-database/photos")
SRC_SKETCH = Path("../cuhk-face-sketch-database/sketches")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "cuhk"

def extract_id(name: str):
    m = re.search(r"([fm])\d*-(\d+-\d+)", name.lower())
    return f"{m.group(1)}-{m.group(2)}" if m else name.lower()

photos = {extract_id(p.stem): p for p in SRC_PHOTO.iterdir() if p.is_file()}
sketches = {extract_id(p.stem): p for p in SRC_SKETCH.iterdir() if p.is_file()}

common_ids = sorted(set(photos.keys()) & set(sketches.keys()))
print(f"成功精准配对: {len(common_ids)} 对 (照片总数={len(photos)}, 素描总数={len(sketches)})")
assert len(common_ids) >= 50, "匹配数量不足 50 对，请检查！"

selected = common_ids[:50]

for split, subset in [("train", selected[:40]), ("test", selected[40:])]:
    p_out = OUT_DIR / split / "photo"
    s_out = OUT_DIR / split / "sketch"
    p_out.mkdir(parents=True, exist_ok=True)
    s_out.mkdir(parents=True, exist_ok=True)

    for idx, pid in enumerate(subset, 1):
        img_p = Image.open(photos[pid]).convert("RGB").resize((256, 256), Image.BICUBIC)
        img_s = Image.open(sketches[pid]).convert("RGB").resize((256, 256), Image.BICUBIC)
        img_p.save(p_out / f"{idx:03d}_{pid}.png")
        img_s.save(s_out / f"{idx:03d}_{pid}.png")

print(f"数据准备完成！位置: {OUT_DIR} (train 40对, test 10对, 256x256 RGB)")

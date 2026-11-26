from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = BASE_DIR / "data" / "cuhk" / "train"


class PairedCUHKDataset(Dataset):
    def __init__(self, root=None, size=256):
        root = Path(root) if root else DEFAULT_ROOT
        self.photo_paths = sorted((root / "photo").glob("*.*"))
        self.sketch_paths = sorted((root / "sketch").glob("*.*"))
        assert len(self.photo_paths) == len(self.sketch_paths) > 0, f"在 {root} 下未找到配对图片！"
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.photo_paths)

    def __getitem__(self, idx):
        ic = self.tf(Image.open(self.photo_paths[idx]).convert("RGB"))
        is_ = self.tf(Image.open(self.sketch_paths[idx]).convert("RGB"))
        return ic, is_


if __name__ == "__main__":
    ds = PairedCUHKDataset(size=256)
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    ic, is_ = next(iter(dl))
    print("训练集样本数:", len(ds))
    print("Ic shape:", ic.shape, "范围:", ic.min().item(), "~", ic.max().item())
    print("Is shape:", is_.shape, "范围:", is_.min().item(), "~", is_.max().item())

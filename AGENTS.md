# AGENTS.md

Guidelines for working in this repository (`StyTR-2` / SAT-StyTR reproduction from `论文.md`).

## Environment & Commands (NixOS, CPU-Only)
- **Always activate `.venv` with `source`** before running Python—`.venv/bin/activate` sets `LD_LIBRARY_PATH` for `nix-ld`. Calling `.venv/bin/python` directly without activating fails with missing shared libraries (e.g. `libz.so.1`).
- Run standalone smoke tests for `scripts/` modules from repo root:
  ```bash
  source .venv/bin/activate && python scripts/dataset_cuhk.py
  source .venv/bin/activate && python scripts/dfdm.py
  source .venv/bin/activate && python scripts/vdfm.py
  source .venv/bin/activate && python scripts/mspd.py
  source .venv/bin/activate && python scripts/sat_losses.py
  source .venv/bin/activate && python scripts/sat_model.py
  ```
- Training / inference / evaluation entrypoints:
  ```bash
  source .venv/bin/activate && python scripts/train_sat.py --steps 500 --batch_size 2
  source .venv/bin/activate && python scripts/infer_sat.py --ckpt out/checkpoints/sat_latest.pth --output out/sat
  source .venv/bin/activate && python scripts/eval_metrics.py --pred_dir out/sat
  ```
- `scripts/prepare-cuhk.py` uses relative `../cuhk-face-sketch-database/{photos,sketches}`, so run it from `scripts/`:
  ```bash
  source .venv/bin/activate && python prepare-cuhk.py   # cwd: scripts/
  ```
- Upstream baseline inference (`test.py` outputs `.jpg` files as `<content>_stylized_<style>.jpg`):
  ```bash
  source .venv/bin/activate && python test.py --content_dir input/content --style_dir input/style --output out
  ```

## Repository Boundaries & Conventions
- **Do not modify** `models/`, `experiments/`, or `cuhk-face-sketch-database/`. Put all new SAT-StyTR modules, losses, training, and evaluation scripts in `scripts/`.
- `test.py`, `models/transformer.py`, `models/ViT_helper.py`, and `util/misc.py` are locally patched for CPU-only execution and newer PyTorch/torchvision compatibility (`torch.device("cpu")`, `map_location='cpu', weights_only=False`). Keep new code CPU-compatible.
- `scripts/` scripts import from `models/` or `scripts/` by inserting `Path(__file__).resolve().parent.parent` into `sys.path`.
- **Optional dependencies**: `skimage`, `lpips`, and `pytorch_fid` are NOT installed in `.venv` (`scripts/eval_metrics.py` computes SSIM/PSNR/MSE in pure PyTorch and skips LPIPS/FID unless installed).
- **Dataset hygiene**:
  - `data/cuhk/` contains 256×256 RGB pairs (`train/`: 40 pairs, `test/`: 10 pairs). Raw CUHK images are 200×250 and fail `PatchEmbed(patch_size=8)` unless resized to a multiple of 8.
  - Never pass `data/cuhk/test/sketch/` as a style input—those are ground-truth targets for evaluation only. Use `data/cuhk/train/sketch/` as style references.
  - Preserve `data/baseline/` as the pre-modification reference; `input/` and `out/` are temporary workspaces kept clean between experiments.
- **Gitignore rules**: `.venv/`, `__pycache__/`, `*.pth`/`*.pt`/`*.ckpt`, `cuhk-face-sketch-database/`, `data/cuhk/`, `/input/`, and `/out/` are ignored. Experimental archives under `data/` (e.g. `data/baseline/`) are tracked.
- **`README.md` policy**: `README.md` is an advisor-facing record containing the full reproduction plan/steps, script descriptions, and result image tables—do not add `[已完成]`/`[待做]` progress status tags to `README.md`.

## Architecture & Paper Mapping Notes
- `models/StyTR.py` provides reusable components: `PatchEmbed` (`[B,3,256,256] -> [B,512,32,32]`), `decoder` (`[B,512,32,32] -> [B,3,256,256]`, deep-copied in `scripts/sat_model.py`), and frozen `vgg` / `enc_1..enc_5`.
- `StyTrans.forward` returns a 5-tuple `(Ics, loss_c, loss_s, loss_lambda1, loss_lambda2)` in its default state (`models/StyTR.py:229`), which `test.py` unwraps via `isinstance(output, tuple)`.
- `test.py` forces `content_size=512`, `style_size=512`, whereas `scripts/` pipelines default to `256×256` (`PairedCUHKDataset(size=256)`).
- When implementing or documenting new `scripts/` components against `论文.md`, map modules/losses explicitly to paper sections/equations/figures and document engineering compromises where the paper omits details (e.g. `mid_ch=64` 1×1 bottleneck in `scripts/dfdm.py`, averaging $L_1/L_2/L_\infty$ with `clamp(max=20.0)` in `scripts/vdfm.py`, 4-layer PatchGAN channels + `AvgPool2d` downsampling in `scripts/mspd.py`, binary edge mask `m` + grayscale $I_c$ in `L_ID1` in `scripts/sat_losses.py`).

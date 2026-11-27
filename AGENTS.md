# AGENTS.md
<!-- 用了一些AI帮助 -->

Guidelines for working in this repository (`StyTR-2` / SAT-StyTR reproduction from `论文.md`).

## Environment & Commands (NixOS, CPU-Only)
- **Always activate `.venv` with `source`** before running Python—`.venv/bin/activate` sets `LD_LIBRARY_PATH` for `nix-ld`. Calling `.venv/bin/python` directly without activating fails with missing shared libraries (e.g. `libz.so.1`).
- After activation, verify `which python` points to this repo's `.venv/bin/python`. If `import torch` fails, check `VIRTUAL_ENV` inside `.venv/bin/activate`; the Cygwin branch still hardcodes the old `StyTR-2` path, while the Linux branch points to `SAT-StyTR`.
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
  source .venv/bin/activate && python scripts/train_sat.py --steps 20 --batch_size 2 --save_every 20 --log_every 5
  source .venv/bin/activate && python scripts/infer_sat.py --ckpt out/checkpoints/sat_latest.pth --output out/sat
  source .venv/bin/activate && python scripts/eval_metrics.py --pred_dir out/sat
  ```
- CPU speed reference: 20 training steps take ~82 s (~4.1 s/step); 500 steps take ~34 min. Do not use the default 120 s command timeout for real training.
- Ablation training uses `--no_dfdm`, `--no_vdfm`, `--no_dfdm --no_vdfm`, `--no_mspd`, `--no_tr_enc`, writing to separate `--save_dir` / `--preview_dir` (see `out/checkpoints_ablation/` and `out/sat_ablation/`).
- `scripts/prepare-cuhk.py` uses relative `../cuhk-face-sketch-database/{photos,sketches}`, so run it from `scripts/`:
  ```bash
  source .venv/bin/activate && python prepare-cuhk.py   # cwd: scripts/
  ```
- Upstream baseline inference (`test.py` outputs `.jpg` files as `<content>_stylized_<style>.jpg`, forced to 512×512):
  ```bash
  source .venv/bin/activate && python test.py --content_dir input/content --style_dir input/style --output out
  source .venv/bin/activate && python test.py --content_dir data/cuhk/test/photo --style data/cuhk/train/sketch/001_f-001-01.png --output out/baseline
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
  - Only the 3 README-referenced example images are force-added from `data/cuhk/`; never add the whole dataset directory.
- **Result archive layout**:
  - `data/baseline/output/` holds the original StyTR-2 baseline outputs. `test.py` generates 512×512 images; archived copies were downsampled to 256×256 so README tables and the comparison matrix use uniform sizes.
  - `data/sat/` holds SAT-StyTR results: `input/`, `style/`, `gt/`, `output/` (full model), `ablation/<variant>/`, and `comparison_grid.png` (the 9×8 README matrix; all cells are 130×130).
  - README-referenced result images must be Git-tracked so they render on GitHub; generated `out/` directories and `*.pth` checkpoints remain ignored.
- **Paper PDF**: the model cannot read PDF directly. Use `pdftotext -layout` for text and `pdftoppm` for figures; the local PDF is reference material and is not tracked unless explicitly requested.
- **Gitignore rules**: `.venv/`, `__pycache__/`, `*.pth`/`*.pt`/`*.ckpt`, `cuhk-face-sketch-database/`, `data/cuhk/`, `/input/`, and `/out/` are ignored. Experimental archives under `data/` (e.g. `data/baseline/`, `data/sat/`) are tracked.
- **`README.md` policy**: `README.md` is an advisor-facing record containing the full reproduction plan/steps, script descriptions, and result image tables—do not add `[已完成]`/`[待做]` progress status tags to `README.md`.

## Reproduction Status & Results
- **Scope**: only the CUHK subset is used—40 training pairs + 10 test pairs (50/188 ≈ 26.6%). The paper uses 150 CUHK training pairs and 38 test pairs; full CUFS has 606 pairs. AR, XM2VTS, and CUFSF were not used.
- **Training budget**: CPU, 20 steps, `batch_size=2`; the paper uses an RTX4090 and 5×10^4 iterations. Current results verify the code path, not paper-level convergence.
- **Conclusion**: the engineering pipeline (DFDM/VDFM/MSPD/losses/generator/train/infer/eval) runs end-to-end, but the paper's quantitative results were **not reproduced**. Full SAT at 20 steps (`SSIM=0.4370`, `PSNR=8.13 dB`, `MSE=10046.92`) is below the original StyTR-2 baseline (`0.5937`, `12.63 dB`, `3643.61`). The best ablation, `w/o DFDM+VDFM`, reaches `0.6071 / 14.11 / 2788.43` but is not the full model.
- **Metrics caveat**: `LPIPS` and `FID` are unavailable because `lpips` / `pytorch-fid` are not installed; do not report them as measured.
- **Visual matrix caveat**: `data/sat/comparison_grid.png` follows the paper's 9-column × 8-row layout. The paper's six external comparison methods (pix2pix, U-GAT-IT, DCLGAN, ref. [15], CAP-VSTNet, Mamba-ST) were not reproduced; columns (b)–(g) are the original baseline plus local ablations. Keep this limitation documented when discussing the matrix.
- **Result artifacts**: full-model checkpoints in `out/checkpoints/`; ablation checkpoints in `out/checkpoints_ablation/<variant>/`; inference outputs in `out/sat/` and `out/sat_ablation/`; archived README assets under `data/sat/`.

## Architecture & Paper Mapping Notes
- `models/StyTR.py` provides reusable components: `PatchEmbed` (`[B,3,256,256] -> [B,512,32,32]`), `decoder` (`[B,512,32,32] -> [B,3,256,256]`, deep-copied in `scripts/sat_model.py`), and frozen `vgg` / `enc_1..enc_5`.
- `StyTrans.forward` returns a 5-tuple `(Ics, loss_c, loss_s, loss_lambda1, loss_lambda2)` in its default state (`models/StyTR.py:229`), which `test.py` unwraps via `isinstance(output, tuple)`.
- `test.py` forces `content_size=512`, `style_size=512`, whereas `scripts/` pipelines default to `256×256` (`PairedCUHKDataset(size=256)`).
- When implementing or documenting new `scripts/` components against `论文.md`, map modules/losses explicitly to paper sections/equations/figures and document engineering compromises where the paper omits details (e.g. `mid_ch=64` 1×1 bottleneck in `scripts/dfdm.py`, averaging $L_1/L_2/L_\infty$ with `clamp(max=20.0)` in `scripts/vdfm.py`, 4-layer PatchGAN channels + `AvgPool2d` downsampling in `scripts/mspd.py`, binary edge mask `m` + grayscale $I_c$ in `L_ID1` in `scripts/sat_losses.py`).
- New SAT-StyTR script-to-paper mapping:
  - `scripts/mspd.py` → 1.4 节 / 图 3
  - `scripts/sat_losses.py` → 1.5 节 / 公式 (9)~(15)
  - `scripts/sat_model.py` → 1.1 节 / 图 1（完整生成器）
  - `scripts/train_sat.py` → 2.1 节（Adam、warmup/decay、消融开关）
  - `scripts/infer_sat.py` → 2.2 节 / 图 4
  - `scripts/eval_metrics.py` → 2.3 节 / 表 1

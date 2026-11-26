# 复现《基于空间自适应 Transformer 的人脸素描图像生成》

## 描述
<!-- 不要更改 -->
使用部分数据复现论文试验方法和结果，复现脚本统一放置于 `scripts/`。

## 环境
<!-- 不要更改 -->
- `venv`
- `Python 3.14.7`
- `CPU`

## 文件结构
<!-- 只显示必要的，参考已经显示的文件结构，尽量简短，禁止更改已存在的注释 -->
```text
.
├── 论文.md                                 # 目标复现论文
├── README.md                               # 复现方法、脚本说明与阶段性测试结果记录
├── scripts                                 # 复现脚本目录
│   ├── prepare-cuhk.py                     # 按人脸编号配对 CUHK 数据并Resize 到 256×256
│   ├── dataset_cuhk.py                     # PyTorch 配对数据读取器 PairedCUHKDataset，返回 (Ic, Is) 张量
│   ├── dfdm.py                             # 双模态特征解耦模块
│   └── vdfm.py                             # 行-列向量动态融合模块
├── cuhk-face-sketch-database               # 原始 CUFS 数据集
├── data
│   ├── cuhk                                # 40 训 / 10 测 迷你数据集
│   │   ├── train
│   │   │   ├── photo                       # 40 张训练照片
│   │   │   └── sketch                      # 40 张训练素描
│   │   └── test
│   │       ├── photo                       # 10 张测试照片
│   │       └── sketch                      # 10 张测试真值素描
│   └── baseline                            # 原版 StyTR-2 在 CUHK 上的基线试验
│       ├── input                           # 基线试验输入
│       └── output                          # 基线生成结果
├── input                                   # 单次推理临时工作区
├── out                                     # 推理输出临时工作区
```

## 步骤
1. 构建 `StyTR-2` 环境，修改代码为 AMD/CPU 可用，使用原版 `test.py` 验证可用性。
2. 下载 CUFS（CUHK）数据集：https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs
   仅使用 `photos/`（188 张，200×250）与 `sketches/`（188 张，200×250）；`cropped_sketch/` 与 `original_sketch/` 属于 CUFSF，暂不使用。
3. 使用 `scripts/prepare-cuhk.py` 从 `photos/` 与 `sketches/` 按人脸编号精准配对，抽取 40 对训练 + 10 对测试（统一 Resize 为 `256×256` RGB），输出到 `data/cuhk/`。
   - 对应论文 **2.1 节**数据集划分（CUHK 188 对按 4:1 轻量化缩减）与 **1.5.4 节公式 (12)** 同人同尺寸约束。
   - 使用原版 `test.py` 跑 1 张测试照片 + 1 张训练素描，输入与输出归档到 `data/baseline/`，对应论文 **2.2 节图 4** 的基线对照组。

   | 输入照片 $I_c$ | 风格参考 $I_s$（训练集） | 原版基线输出（`data/baseline/output/`） | 测试真值素描（仅对比） |
   | :---: | :---: | :---: | :---: |
   | ![Ic](data/baseline/input/content/001_f-041-01.png) | ![Is](data/baseline/input/style/001_f-001-01.png) | ![baseline](data/baseline/output/001_f-041-01_stylized_001_f-001-01.jpg) | ![gt](data/cuhk/test/sketch/001_f-041-01.png) |

4. 使用 `scripts/dataset_cuhk.py` 构建配对数据读取器 `PairedCUHKDataset`（输出张量 `[B, 3, 256, 256]`，像素范围 `[0, 1]`）。
   - 对应论文 **1.1 节**双分支输入（人脸照片 $I_c$、素描图像 $I_s$）与 **1.5.4 节公式 (12)** 像素对齐要求。

   | 训练照片 $I_c$ | 对应训练素描 $I_s$ |
   | :---: | :---: |
   | ![train_photo](data/cuhk/train/photo/001_f-001-01.png) | ![train_sketch](data/cuhk/train/sketch/001_f-001-01.png) |

5. 使用 `scripts/dfdm.py` 实现并测通 `DFDM×3`：内容分支 3 级最大池化（`3×3→2×2`）、风格分支 5 级渐进平均池化（`5×5→3×3`）、`13×13` 与 `3×3` 交替卷积、双线性插值回原尺寸后逐像素重加权，参数量 `8.95 M`。
   - 对应论文 **1.2 节**、**图 2**、**2.4.2 节**与 **2.4.3 节**。
6. 使用 `scripts/vdfm.py` 实现并测通 `VDFM×3`：对内容与风格特征通道拼接后做自注意力与线性投影，沿行/列做 $L_1/L_2/L_\infty$ 多范数聚合与指数门控双向融合，参数量 `14.96 M`，串联 `PatchEmbed -> DFDM×3 -> VDFM×3 -> decoder` 输出 `[1, 3, 256, 256]` 并完成梯度反传验证。
   - 对应论文 **1.3 节公式 (1)~(8)** 与 **图 1**。

## 原版 StyTR-2 已实现的论文部分
- `models/StyTR.py` (14~33 行) `PatchEmbed`：分块卷积嵌入（`patch_size=8`，`256×256 → 32×32×512`）→ 对应 **1.1 节**“分块卷积操作”与 **图 1**“嵌入”。
- `models/transformer.py` (48~50 行) `Transformer.forward`：内容自适应位置嵌入 → 对应 **图 1**“位置嵌入”。
- `models/transformer.py` (23~24 行) `encoder_c` / `encoder_s`：3 层双路自注意力编码器 → 对应 **图 1** 编码端的层归一化与自注意力骨架（层数与 `DFDM×3` 一致）。
- `models/StyTR.py` (36~66 行) `decoder`：3 次上采样卷积重建图像 → 对应 **1.1 节**轻量化解码器与 **图 1** 右侧重建端。
- `models/StyTR.py` (68~165 行) `vgg` / `enc_1..enc_5`：冻结的预训练 VGG19 特征提取 $\Phi(\cdot)$ → 对应 **1.5 节**特征空间约束。
- `models/StyTR.py` (209 行) `calc_content_loss`：VGG 第 4、5 层特征 MSE → 对应 **1.5.1 节公式 (9)** $L_C$（论文权重 7）。
- `models/StyTR.py` (211~213 行) `calc_style_loss`：5 层 VGG 均值 + 标准差 MSE → 对应 **1.5.2 节公式 (10)** $L_S$（论文权重 18）。
- `models/StyTR.py` (216~227 行) `loss_lambda1` / `loss_lambda2`：自重建身份损失（权重 70 与 1，与论文 **1.5.4 节公式 (12)(13)** 的 $\lambda_{ID1}=70$、$\lambda_{ID2}=1$ 一致，区别是论文改为直接约束生成图 $G_s$ 与配对 $I_c, I_s$）。

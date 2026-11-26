# 术语与符号缩写表（ALIAS）

整理自目标论文《基于空间自适应 Transformer 的人脸素描图像生成》（`论文.md`）、原版 `StyTR-2` 源码及 `scripts/` 复现代码。

---

## 1. 论文核心模型与模块（1.1 ~ 1.4 节）

| 缩写 / 符号 | 全称 | 中文 | 含义 / 出现位置 |
|---|---|---|---|
| **SAT-StyTR** | Spatially Adaptive Transformer for Face Photo-to-Sketch Synthesis | 基于空间自适应 Transformer 的人脸素描生成网络 | 本文所提半监督 GAN 总框架；论文 1.1 节、图 1 |
| **StyTR-2 / $\text{StyTr}^2$** | Style Transfer with Transformers | 基于 Transformer 的风格迁移网络 | 本复现的底座模型（原版仓库）；论文引言 [16]、`models/StyTR.py` |
| **DFDM** | Dual-modal Feature Disentanglement Module | 双模态特征解耦模块 | 并行内容最大池化分支 + 风格平均池化分支，解耦语义与风格；论文 1.2 节、图 2、`scripts/dfdm.py` |
| **VDFM** | Vector Dynamic Fusion Module | 行-列向量动态融合模块 | 拼接自注意力 + 行/列多范数双向门控融合；论文 1.3 节公式 (1)~(8)、`scripts/vdfm.py` |
| **MSPD** | Multi-scale Perceptive Discriminator | 多尺度感知判别器 | 3 尺度特征金字塔判别器，同时约束全局结构与局部笔触；论文 1.4 节、图 3、`scripts/mspd.py` |
| **SA** | Self-Attention | 自注意力机制 | 建模特征图长程空间依赖；论文 1.3 节公式 (1)、`nn.MultiheadAttention` |
| **LoG** | Laplacian of Gaussian | 拉普拉斯-高斯算子 | 先高斯平滑降噪再用拉普拉斯提取二阶高频边缘；论文 1.5.3 节公式 (11) |
| **GAN** | Generative Adversarial Network | 生成对抗网络 | 生成器 $G$ 与判别器 $D$ 博弈优化；论文 1.1 / 1.4 / 1.5.5 节 |
| **CNN** | Convolutional Neural Network | 卷积神经网络 | 局部卷积特征提取网络；论文 0 引言 |

---

## 2. 数据集（2.1 节）

| 缩写 / 符号 | 全称 | 中文 | 含义 / 出现位置 |
|---|---|---|---|
| **CUFS** | CUHK Face Sketch Database | 香港中文大学人脸素描数据集 | 共 606 对照片-素描，由 CUHK、AR、XM2VTS 组成；论文 2.1 节 [20] |
| **CUHK** | The Chinese University of Hong Kong (subset) | 香港中文大学子集 | CUFS 中 188 对学生人脸数据；本复现从中取 40 训 / 10 测，见 `data/cuhk/` |
| **AR** | Aleix & Robert Face Database | AR 人脸子集 | CUFS 子集（123 对，100 训 / 23 测）；论文 2.1 节 |
| **XM2VTS** | Extended Multi Modal Verification for Teleservices and Security | XM2VTS 人脸子集 | CUFS 子集（295 对，100 训 / 195 测）；论文 2.1 节 |
| **CUFSF** | CUHK Face Sketch FERET Database | CUFSF 素描数据集 | 含复杂光照/夸张轮廓的 1194 对数据（999 训 / 195 测），暂未使用；论文 2.1 节 [21] |

---

## 3. 损失函数与数学变量（1.2 ~ 1.5 节，公式 1 ~ 15）

| 缩写 / 符号 | 全称 / 读法 | 中文 | 含义 / 出现位置 |
|---|---|---|---|
| **$I_c$** | Input Content photo | 输入人脸照片 | 内容分支输入（提供五官轮廓与身份结构）；论文 1.5 节、`scripts/dataset_cuhk.py` |
| **$I_s$** | Input Style sketch | 输入素描图像 | 风格分支参考图（训练/评估中亦指配对真值素描）；论文 1.5 节 |
| **$G_s$ / $G(I_c)$** | Generated Sketch | 生成素描图像 | 生成器输出的人脸素描图；论文 1.5 节 |
| **$X_1, X_2$** | Content / Style Features | 解耦后的内容特征 / 风格特征 | DFDM×3 输出送入 VDFM 的两组特征张量；论文 1.3 节公式 (1)(2) |
| **$X_f, X_2'$** | Aligned Fused / Projected Style Feature | 对齐融合特征 / 投影风格特征 | 分别由 $\text{Linear}(\text{SA}(\text{Concat}(X_1, X_2)))$ 与 $\text{Linear}(X_2)$ 得到；公式 (1)(2) |
| **$A_i(\cdot)$** | Row/Column Attention ($i=1,2$) | 行向量 / 列向量注意力操作 | 沿行（$i=1$）或列（$i=2$）做 $\overline{L}_1 / \overline{L}_2 / L_\infty$ 聚合；公式 (3) |
| **$\overline{L}_1, \overline{L}_2, L_\infty$** | Normalized $L_1, L_2$, Max Norm | 归一化 $L_1/L_2$ 范数与无穷范数 | 衡量向量平均响应强度、能量分布与极值边缘；公式 (3)、`axis_norm()` |
| **$\alpha_i, \beta_i$** | Bidirectional Gating Weights | 双向门控权重系数 | 经指数增强归一化后分配给 $X_f$ 与 $X_2'$ 的融合权重；公式 (4)(5) |
| **$F_1, F_2, F$** | Row / Column / Final Fused Tensor | 行融合 / 列融合 / 最终融合特征 | $F_1, F_2$ 分别按行/列加权融合，$F = F_1 + F_2$；公式 (6)(7)(8) |
| **$D_i(\cdot)$** | $i$-th Scale Discriminator ($i=1,2,3$) | 第 $i$ 尺度子判别器 | MSPD 三个分辨率分支的真假判别概率输出；论文 1.4 节、公式 (14) |
| **$\Phi(\cdot)$ / $\Phi_i(\cdot)$** | VGG19 Feature Extractor | VGG19 特征提取操作 | 冻结的预训练 VGG19 第 $i$ 层特征映射；论文 1.5 节、`models/StyTR.py` `enc_1..enc_5` |
| **$\mu(\cdot), \sigma(\cdot)$** | Channel Mean & Standard Deviation | 通道均值与标准差 | 表征素描风格分布的统计量；公式 (10)、`function.py` `calc_mean_std` |
| **$\text{lap}(\cdot), f, m$** | Laplacian, Gaussian filter, Edge mask | 拉普拉斯算子、高斯滤波、边缘掩码 | 组成 $L_D$ 的高频边缘提取与噪声抑制项；公式 (11) |
| **$L_C$** | Content Loss | 语义内容损失 | VGG19 深层特征 L2 距离（论文权重 $\lambda_C = 7$）；公式 (9) |
| **$L_S$** | Style Loss | 风格分布损失 | 通道均值与标准差 L2 距离（论文权重 $\lambda_S = 18$）；公式 (10) |
| **$L_D$** | Detail Loss (LoG) | 微观细节损失 | 拉普拉斯-高斯高频边缘 L2 距离（论文权重 $\lambda_D = 200$）；公式 (11) |
| **$L_{ID1}$** | Pixel Identity Loss | 像素级身份损失 | $G_s$ 与 $I_c, I_s$ 的像素级 L2 约束（论文权重 $\lambda_{ID1} = 70$）；公式 (12) |
| **$L_{ID2}$** | Perceptual Identity Loss | 深度特征身份损失 | VGG19 前 5 层多尺度特征 MSE（论文权重 $\lambda_{ID2} = 1$）；公式 (13) |
| **$L_A$** | Multi-scale Adversarial Loss | 多尺度对抗损失 | 3 尺度判别器 $-\ln D_i(G(I_c))$ 平均（论文权重 $\lambda_A = 1$）；公式 (14) |
| **$L$** | Total Weighted Loss | 总加权损失 | $L = \lambda_C L_C + \lambda_S L_S + \lambda_D L_D + \lambda_{ID1} L_{ID1} + \lambda_{ID2} L_{ID2} + \lambda_A L_A$；公式 (15) |
| **$\lambda_C, \lambda_S, \dots, \lambda_A$** | Loss Weights | 损失权重系数 | 论文设定为 $7, 18, 200, 70, 1, 1$；公式 (15) |
| **$\epsilon$ (`eps`)** | Numerical Stability Constant | 防除零小常数 | 避免方差开根或 Softmax 分母为 0（代码常用 `1e-5` 或 `1e-6`）；公式 (4)(5) |

---

## 4. 评价指标（2.3 节，表 1）

| 缩写 / 符号 | 全称 | 中文 | 含义 / 方向 |
|---|---|---|---|
| **SSIM** | Structural Similarity Index Measure | 结构相似度 | 衡量亮度、对比度与结构保真度；**越大越好（↑）** |
| **PSNR** | Peak Signal-to-Noise Ratio | 峰值信噪比（单位 dB） | 基于像素误差的对数信噪比；**越大越好（↑）** |
| **MSE** | Mean Squared Error | 均方误差 | 生成图与真值图逐像素平方差均值；**越小越好（↓）** |
| **LPIPS** | Learned Perceptual Image Patch Similarity | 学习感知图像块相似度 | 基于深度网络特征的人眼感知距离；**越小越好（↓）** |
| **FID** | Fréchet Inception Distance | 弗雷歇 Inception 距离 | 生成分布与真实素描分布的总体特征距离；**越小越好（↓）** |

---

## 5. 训练策略与效率指标（2.1 / 2.6 节）

| 缩写 / 符号 | 全称 | 中文 | 含义 / 出现位置 |
|---|---|---|---|
| **Adam** | Adaptive Moment Estimation | Adam 自适应优化器 | 生成器与判别器训练优化器；论文 2.1 节 [22] |
| **lr** | Learning Rate | 学习率 | 论文设置生成器 $8\times10^{-4}$、判别器 $2\times10^{-4}$；2.1 节 |
| **warmup** | Linear Warmup | 线性预热 | 训练前期（论文 $< 10^4$ iter）学习率线性上升以稳住初始梯度；2.1 节 |
| **decay** | Exponential Decay | 指数衰减 | 训练中后期（论文 $\ge 10^4$ iter）学习率指数下降促进收敛；2.1 节 |
| **iter / step** | Iteration | 迭代步数 | 一次参数更新为一个 iter（论文共 $5\times10^4$ iter）；2.1 节 |
| **$N_p$** | Number of Parameters | 模型参数量（单位 M = $10^6$） | 论文 SAT-StyTR 总参数量 83.74 M；表 5 |
| **$t_{\text{inf}}$** | Average Inference Time | 平均推理耗时（单位 s） | 单张图前向生成耗时（论文 0.041 s）；表 5 |
| **w/o / w/** | without / with | 不含 / 包含 | 消融实验记号（如 `w/o branch` vs `w/ branch`）；表 3 |

---

## 6. 网络结构与 PyTorch 代码组件

| 缩写 / 符号 | 全称 | 中文 | 含义 / 出现位置 |
|---|---|---|---|
| **VGG19** | Visual Geometry Group 19-layer Network | 19 层 VGG 特征网络 | 预训练冻结特征提取器与多尺度损失骨干；`models/StyTR.py` |
| **ViT** | Vision Transformer | 视觉 Transformer | 将图像分块展平送入 Transformer；`models/ViT_helper.py` |
| **PatchEmbed** | Patch Embedding | 分块卷积嵌入 | 用 `8×8` 步长 8 卷积将 `[B,3,256,256]` 变为 `[B,512,32,32]`；`models/StyTR.py` |
| **MLP / FFN** | Multi-Layer Perceptron / Feed-Forward Network | 多层感知机 / 前馈网络 | Transformer 编码/解码器中的全连接变换层；`models/StyTR.py` |
| **MaxPool / AvgPool** | Max Pooling / Average Pooling | 最大池化 / 平均池化 | DFDM 内容分支（留边缘）与风格分支（平滑笔触）的异构池化；`scripts/dfdm.py` |
| **ReflectionPad2d** | 2D Reflection Padding | 二维镜像填充 | 卷积前镜像补边以减少边缘伪影；`models/StyTR.py` `decoder` |
| **Interpolate** | Bilinear / Nearest Interpolation | 双线性 / 最近邻插值 | 将缩小的注意力图或特征图放大回原尺寸；`scripts/dfdm.py`、`decoder` |
| **Sigmoid** | Sigmoid Activation $\sigma(x)=\frac{1}{1+e^{-x}}$ | Sigmoid 激活函数 | 将输出压缩到 $(0,1)$ 区间，用于像素注意力图与判别器概率输出 |
| **NestedTensor** | Nested Tensor `(tensor, mask)` | 带掩码嵌套张量 | DETR/StyTR-2 遗留封装，打包图像张量与 padding 掩码；`util/misc.py` |
| **`normal()` / `calc_mean_std`** | Feature Normalization / Channel Stats | 特征归一化 / 均值方差计算 | 计算通道均值方差及归一化特征用于 $L_C, L_S$；`function.py` |

---

## 7. 环境、张量维度与目录术语

| 缩写 / 符号 | 全称 | 中文 | 含义 / 出现位置 |
|---|---|---|---|
| **$B, C, H, W$** | Batch, Channels, Height, Width | 批大小、通道数、高、宽 | PyTorch 图像/特征张量维度惯例（如 `[B, 3, 256, 256]`、`[B, 512, 32, 32]`） |
| **RGB** | Red, Green, Blue | 三通道彩色图像格式 | 素描原图虽为灰度，代码统一 `.convert("RGB")` 成 3 通道以对齐 $I_c$ |
| **`.venv`** | Python Virtual Environment | Python 虚拟环境 | 本仓库 NixOS 环境，须 `source .venv/bin/activate` 加载 `LD_LIBRARY_PATH` |
| **`train` / `test`** | Training Set / Test Set | 训练集 / 测试集 | `data/cuhk/train/`（40 对训练）与 `data/cuhk/test/`（10 对考核） |
| **`photo` / `sketch`** | Face Photo / Face Sketch | 人脸照片 / 人脸素描 | `data/cuhk/{train,test}/` 下两种配对图像子目录 |
| **`content` / `style`** | Content Input / Style Reference | 待迁移内容图 / 风格参考图 | `input/`、`data/baseline/input/` 下的推理输入子目录 |
| **`baseline`** | Baseline | 基线（改造前对照） | `data/baseline/` 存放原版 StyTR-2 未改模块前的输入与输出对照图 |
| **`experiments`** | Experiments (Pretrained Checkpoints) | 预训练权重目录 | 存放原版 4 个 `.pth` 权重文件 |

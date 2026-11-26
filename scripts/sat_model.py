import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.StyTR as StyTR
import models.transformer as tr
from scripts.dfdm import DFDM
from scripts.vdfm import VDFM

BASE_DIR = Path(__file__).resolve().parent.parent


class SATStyTRGenerator(nn.Module):
    """
    对应论文 1.1~1.3 节 & 图 1:
    PatchEmbed -> DFDM×3 -> (Transformer 双路编码器 + 位置嵌入) -> VDFM×3 -> Decoder
    支持消融开关: use_dfdm / use_vdfm / use_transformer_encoder
    """
    def __init__(
        self,
        img_size=256,
        embed_dim=512,
        mid_ch=64,
        num_heads=8,
        use_dfdm=True,
        use_vdfm=True,
        use_transformer_encoder=True,
    ):
        super().__init__()
        self.use_dfdm = use_dfdm
        self.use_vdfm = use_vdfm
        self.use_transformer_encoder = use_transformer_encoder

        self.embedding = StyTR.PatchEmbed(img_size=img_size, patch_size=8, in_chans=3, embed_dim=embed_dim)
        self.decoder = copy.deepcopy(StyTR.decoder)

        if self.use_dfdm:
            self.dfdm_blocks = nn.ModuleList([DFDM(embed_dim, mid_ch=mid_ch) for _ in range(3)])

        if self.use_transformer_encoder:
            base_tr = tr.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=3, num_decoder_layers=3)
            self.encoder_c = base_tr.encoder_c
            self.encoder_s = base_tr.encoder_s
            self.new_ps = base_tr.new_ps
            self.averagepooling = base_tr.averagepooling

        if self.use_vdfm:
            self.vdfm_blocks = nn.ModuleList([VDFM(embed_dim, num_heads=num_heads) for _ in range(3)])

    def load_pretrained_stytr(self, exp_dir=None):
        """加载原版 StyTR-2 预训练权重作为微调起点（嵌入层、双路编码器、位置嵌入、解码器）"""
        exp_dir = Path(exp_dir) if exp_dir else (BASE_DIR / "experiments")
        emb_path = exp_dir / "embedding_iter_160000.pth"
        dec_path = exp_dir / "decoder_iter_160000.pth"
        tr_path = exp_dir / "transformer_iter_160000.pth"

        if emb_path.exists():
            self.embedding.load_state_dict(torch.load(emb_path, map_location="cpu", weights_only=False))
        if dec_path.exists():
            self.decoder.load_state_dict(torch.load(dec_path, map_location="cpu", weights_only=False))
        if self.use_transformer_encoder and tr_path.exists():
            sd = torch.load(tr_path, map_location="cpu", weights_only=False)
            enc_c_sd = {k[len("encoder_c."):]: v for k, v in sd.items() if k.startswith("encoder_c.")}
            enc_s_sd = {k[len("encoder_s."):]: v for k, v in sd.items() if k.startswith("encoder_s.")}
            ps_sd = {k[len("new_ps."):]: v for k, v in sd.items() if k.startswith("new_ps.")}
            self.encoder_c.load_state_dict(enc_c_sd)
            self.encoder_s.load_state_dict(enc_s_sd)
            self.new_ps.load_state_dict(ps_sd)

    def forward(self, ic, is_):
        feat_c = self.embedding(ic)
        feat_s = self.embedding(is_)

        if self.use_dfdm:
            for blk in self.dfdm_blocks:
                feat_c, feat_s = blk(feat_c, feat_s)

        if self.use_transformer_encoder:
            B, C, H, W = feat_c.shape
            pos_c = self.new_ps(self.averagepooling(feat_c))
            pos_embed_c = F.interpolate(pos_c, mode="bilinear", size=(H, W), align_corners=False)

            seq_c = feat_c.flatten(2).permute(2, 0, 1)
            seq_s = feat_s.flatten(2).permute(2, 0, 1)
            pos_seq_c = pos_embed_c.flatten(2).permute(2, 0, 1)

            seq_c = self.encoder_c(seq_c, pos=pos_seq_c)
            seq_s = self.encoder_s(seq_s)

            feat_c = seq_c.permute(1, 2, 0).view(B, C, H, W)
            feat_s = seq_s.permute(1, 2, 0).view(B, C, H, W)

        if self.use_vdfm:
            fused = feat_c
            for blk in self.vdfm_blocks:
                fused = blk(fused, feat_s)
        else:
            fused = 0.5 * (feat_c + feat_s)

        return self.decoder(fused)


if __name__ == "__main__":
    gen = SATStyTRGenerator()
    gen.load_pretrained_stytr()
    params = sum(p.numel() for p in gen.parameters()) / 1e6
    print(f"SAT-StyTR 生成器总参数量: {params:.2f} M")

    ic = torch.rand(1, 3, 256, 256)
    is_ = torch.rand(1, 3, 256, 256)
    gs = gen(ic, is_)
    print("输出 Gs 形状:", gs.shape)
    gs.mean().backward()
    print("SATStyTRGenerator 预训练权重加载 + 前向/反传成功!")

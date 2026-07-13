import torch
import torch.nn as nn


class CoAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.img2txt_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.txt2img_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.img_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.txt_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_img_cross = nn.LayerNorm(d_model)
        self.norm_txt_cross = nn.LayerNorm(d_model)
        self.norm_img_self = nn.LayerNorm(d_model)
        self.norm_txt_self = nn.LayerNorm(d_model)
        self.norm_img_ffn = nn.LayerNorm(d_model)
        self.norm_txt_ffn = nn.LayerNorm(d_model)

        self.img_ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.txt_ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_tokens, txt_tokens, txt_key_padding_mask=None):

        # 1. image queries attend to text keys/values
        img_cross, img2txt_w = self.img2txt_attn(
            query=img_tokens, key=txt_tokens, value=txt_tokens,
            key_padding_mask=txt_key_padding_mask,
        )
        img_tokens = self.norm_img_cross(img_tokens + self.dropout(img_cross))

        # 2. text queries attend to image keys/values (image has no padding)
        txt_cross, txt2img_w = self.txt2img_attn(
            query=txt_tokens, key=img_tokens, value=img_tokens,
        )
        txt_tokens = self.norm_txt_cross(txt_tokens + self.dropout(txt_cross))

        # 3. self-attention within each modality, now cross-informed
        img_self, _ = self.img_self_attn(img_tokens, img_tokens, img_tokens)
        img_tokens = self.norm_img_self(img_tokens + self.dropout(img_self))

        txt_self, _ = self.txt_self_attn(
            txt_tokens, txt_tokens, txt_tokens, key_padding_mask=txt_key_padding_mask,
        )
        txt_tokens = self.norm_txt_self(txt_tokens + self.dropout(txt_self))

        # 4. per-modality feed-forward
        img_tokens = self.norm_img_ffn(img_tokens + self.img_ffn(img_tokens))
        txt_tokens = self.norm_txt_ffn(txt_tokens + self.txt_ffn(txt_tokens))

        attn_weights = {"img2txt": img2txt_w, "txt2img": txt2img_w}
        return img_tokens, txt_tokens, attn_weights


class CoAttentionEncoder(nn.Module):
    """Stack of N CoAttentionLayers."""

    def __init__(self, d_model, n_heads, ff_dim, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CoAttentionLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])

    def forward(self, img_tokens, txt_tokens, txt_key_padding_mask=None):
        all_attn_weights = []
        for layer in self.layers:
            img_tokens, txt_tokens, attn_weights = layer(
                img_tokens, txt_tokens, txt_key_padding_mask
            )
            all_attn_weights.append(attn_weights)
        return img_tokens, txt_tokens, all_attn_weights

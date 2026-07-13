"""
SkinScanFusionModel — end-to-end cross-attention multimodal fusion.

Pipeline:
  image  -> CVSpatialFeatureExtractor   -> image_tokens  (B, 49,  512-ish)
  text   -> BioBERTTokenExtractor       -> text_tokens   (B, L,   768)
                                            + attention_mask
                     |
                     v
       project both to a shared d_model
                     |
                     v
          CoAttentionEncoder (N layers)
                     |
        masked mean-pool each modality
                     |
                     v
   learned gate blends image/text pooled vectors
                     |
                     v
   [img_pooled | txt_pooled | gated_fusion] -> MLP -> fused logits (10 classes)

Also returns per-modality auxiliary logits (image-only, text-only) computed
from the *pre-fusion* pooled embeddings CV/NLP already produce. Training on
fused + aux losses together keeps a gradient signal in each branch even
while the paired multimodal dataset is small — and it means the fusion
model gracefully degrades: if the image is blurry/ambiguous, the model still
has a usable text-only signal to lean on, and vice versa.
"""

import torch
import torch.nn as nn

import config
from cross_attention import CoAttentionEncoder


class SkinScanFusionModel(nn.Module):
    def __init__(
        self,
        cv_extractor,
        nlp_extractor,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        ff_dim=config.FF_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES,
    ):
        super().__init__()
        self.cv_extractor = cv_extractor
        self.nlp_extractor = nlp_extractor

        img_channels = cv_extractor.spatial_channels
        txt_hidden = nlp_extractor.hidden_size

        self.img_proj = nn.Linear(img_channels, d_model)
        self.txt_proj = nn.Linear(txt_hidden, d_model)

        
        num_patches = getattr(cv_extractor, "num_patches", 49)  
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.img_pos_embed, std=0.02)

        self.co_attention = CoAttentionEncoder(d_model, n_heads, ff_dim, n_layers, dropout)

       
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.fused_classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )


        self.image_aux_classifier = nn.Linear(config.CV_EMBEDDING_DIM, num_classes)
        self.text_aux_classifier = nn.Linear(config.NLP_EMBEDDING_DIM, num_classes)

    @staticmethod
    def _masked_mean(tokens, key_padding_mask):
        """key_padding_mask: True = PAD. tokens: (B, L, D)."""
        keep_mask = (~key_padding_mask).unsqueeze(-1).float()  # (B, L, 1)
        summed = (tokens * keep_mask).sum(dim=1)
        counts = keep_mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, images, texts, return_attention=False):

        image_tokens_raw, image_pooled = self.cv_extractor(images)          # (B,N,C_img), (B,512)
        text_tokens_raw, attention_mask, text_pooled = self.nlp_extractor(texts)  # (B,L,768), (B,L), (B,768)
        txt_key_padding_mask = attention_mask == 0  # True where PAD

        img_tokens = self.img_proj(image_tokens_raw) + self.img_pos_embed[:, : image_tokens_raw.size(1), :]
        txt_tokens = self.txt_proj(text_tokens_raw)

        img_tokens, txt_tokens, attn_weights = self.co_attention(
            img_tokens, txt_tokens, txt_key_padding_mask
        )

        img_fused_pooled = img_tokens.mean(dim=1)                              # (B, d_model)
        txt_fused_pooled = self._masked_mean(txt_tokens, txt_key_padding_mask)  # (B, d_model)

        gate_input = torch.cat([img_fused_pooled, txt_fused_pooled], dim=-1)
        g = self.gate(gate_input)                                              # (B, 1)
        gated_fusion = g * img_fused_pooled + (1 - g) * txt_fused_pooled

        classifier_input = torch.cat(
            [img_fused_pooled, txt_fused_pooled, gated_fusion], dim=-1
        )
        fused_logits = self.fused_classifier(classifier_input)

        image_aux_logits = self.image_aux_classifier(image_pooled)
        text_aux_logits = self.text_aux_classifier(text_pooled)

        output = {
            "fused_logits": fused_logits,
            "image_aux_logits": image_aux_logits,
            "text_aux_logits": text_aux_logits,
            "fused_probs": torch.softmax(fused_logits, dim=-1),
            "gate": g.squeeze(-1),  # how image-leaning (near 1) vs text-leaning (near 0) each example was
        }
        if return_attention:
            output["attn_weights"] = attn_weights
        return output


def compute_loss(output, labels, criterion=None):

    criterion = criterion or nn.CrossEntropyLoss()
    loss = (
        config.FUSED_LOSS_WEIGHT * criterion(output["fused_logits"], labels)
        + config.AUX_IMAGE_LOSS_WEIGHT * criterion(output["image_aux_logits"], labels)
        + config.AUX_TEXT_LOSS_WEIGHT * criterion(output["text_aux_logits"], labels)
    )
    return loss

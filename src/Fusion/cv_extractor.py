import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
)

import config


class SkinLesionClassifier(nn.Module):
    """Exact mirror of CV/model.py — see that file for authoritative docs."""

    def __init__(
        self,
        model_name=config.CV_BACKBONE,
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.CV_EMBEDDING_DIM,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        self.model_name = model_name.lower()

        if self.model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            self.features = backbone.features
            self.pool = backbone.avgpool
            
            self._last_conv_layer = self.features[-1]
        elif self.model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
            in_features = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self._last_conv_layer = backbone.layer4[-1]
        else:
            raise ValueError(f"Unknown model_name '{model_name}'")

        self.spatial_channels = in_features  

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.embedding_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        feats = self.features(x)
        pooled = torch.flatten(self.pool(feats), 1)
        embedding = self.embedding_head(pooled)
        logits = self.classifier_head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


class CVSpatialFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint_path=config.CV_CHECKPOINT_PATH,
        backbone=config.CV_BACKBONE,
        freeze=config.CV_FREEZE_BACKBONE,
        unfreeze_last_n_blocks=config.CV_UNFREEZE_LAST_N_BLOCKS,
        device=config.DEVICE,
    ):
        super().__init__()
        import os
        checkpoint_exists = bool(checkpoint_path) and os.path.exists(checkpoint_path)

        
        self.classifier = SkinLesionClassifier(model_name=backbone, pretrained=not checkpoint_exists)

        if checkpoint_exists:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.classifier.load_state_dict(state_dict)
            print(f"[CV] loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[CV] WARNING: checkpoint not found at '{checkpoint_path}'. "
                  f"Using ImageNet-pretrained backbone only (fine for a shape/wiring "
                  f"smoke test, NOT for real training/inference).")

        if freeze:
            for p in self.classifier.features.parameters():
                p.requires_grad = False
            if unfreeze_last_n_blocks > 0:
                blocks = list(self.classifier.features.children())
                for block in blocks[-unfreeze_last_n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True

        self.spatial_channels = self.classifier.spatial_channels

    def forward(self, images):
        
        feats = self.classifier.features(images)           # (B, C, H', W')
        B, C, H, W = feats.shape
        image_tokens = feats.flatten(2).transpose(1, 2)     # (B, H'*W', C)

        pooled = torch.flatten(self.classifier.pool(feats), 1)
        pooled_embedding = self.classifier.embedding_head(pooled)  # (B, embedding_dim)

        return image_tokens, pooled_embedding

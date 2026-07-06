"""
Skin lesion classifier — supports multiple backbones (EfficientNet-B0, ResNet50).

Which backbone gets built is controlled by config.MODEL_NAME:
    config.MODEL_NAME = "efficientnet_b0"   (default, fast/light)
    config.MODEL_NAME = "resnet50"          (heavier, sometimes stronger features)

Design goal: the model must expose BOTH
  - class logits (for training / evaluation)
  - a fixed-size embedding vector (for the Fusion member, member 5)

So predict.py can return:
{
    "class": "Eczema",
    "confidence": 0.94,
    "embedding": [...]   # config.EMBEDDING_DIM floats
}

Note: the embedding_dim is always config.EMBEDDING_DIM (512) regardless of
backbone, so Fusion (member 5) never has to care which backbone CV used.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
)

import config


class SkinLesionClassifier(nn.Module):
    def __init__(
        self,
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE,
    ):
        super().__init__()

        self.model_name = model_name.lower()

        if self.model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)

            # EfficientNet-B0 head is: Dropout -> Linear(1280, 1000)
            in_features = backbone.classifier[1].in_features  # 1280

            self.features = backbone.features
            self.pool = backbone.avgpool  # AdaptiveAvgPool2d(1)
            self._last_conv_layer = self.features[-1]

        elif self.model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)

            in_features = backbone.fc.in_features  # 2048

            # ResNet50's own children up to (but not including) avgpool/fc
            # are exactly the conv feature extractor we need.
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            # backbone.layer4 is the last conv block, good Grad-CAM target
            self._last_conv_layer = backbone.layer4[-1]

        else:
            raise ValueError(
                f"Unknown MODEL_NAME '{model_name}'. "
                f"Supported: 'efficientnet_b0', 'resnet50'."
            )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Embedding head: this is what Fusion (member 5) consumes.
        # Same shape (embedding_dim) no matter which backbone is used.
        self.embedding_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Classification head: sits on top of the embedding.
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        feats = self.features(x)          # (B, C, H, W) - C depends on backbone
        pooled = self.pool(feats)          # (B, C, 1, 1)
        pooled = torch.flatten(pooled, 1)  # (B, C)

        embedding = self.embedding_head(pooled)   # (B, embedding_dim)
        logits = self.classifier_head(embedding)  # (B, num_classes)

        if return_embedding:
            return logits, embedding
        return logits

    def get_last_conv_layer(self):
        """
        Convenience for Grad-CAM (member 4 / Explainability).
        Returns the last convolutional layer, the typical Grad-CAM target,
        regardless of which backbone is active.
        """
        return self._last_conv_layer


def build_model():
    model = SkinLesionClassifier()
    print(f"Built model with backbone: {model.model_name}")
    return model.to(config.DEVICE)


if __name__ == "__main__":
    model = build_model()
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)

    logits, embedding = model(dummy_input, return_embedding=True)
    print("Logits shape:", logits.shape)         # torch.Size([2, NUM_CLASSES])
    print("Embedding shape:", embedding.shape)   # torch.Size([2, EMBEDDING_DIM])

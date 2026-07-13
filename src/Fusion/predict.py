import torch
from PIL import Image
from torchvision import transforms

import config
from cv_extractor import CVSpatialFeatureExtractor
from nlp_extractor import BioBERTTokenExtractor
from fusion_model import SkinScanFusionModel


class FusionPredictor:
    def __init__(self, fusion_checkpoint_path, device=config.DEVICE):
        self.device = device
        cv_extractor = CVSpatialFeatureExtractor(freeze=True).to(device)
        nlp_extractor = BioBERTTokenExtractor(freeze=True).to(device)
        self.model = SkinScanFusionModel(cv_extractor, nlp_extractor).to(device)

        state_dict = torch.load(fusion_checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.image_transform = transforms.Compose([
            transforms.Resize((config.CV_IMAGE_SIZE, config.CV_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image_path, symptom_text, return_attention=False):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        output = self.model(image_tensor, [symptom_text], return_attention=return_attention)

        probs = output["fused_probs"].squeeze(0)
        top_idx = probs.argmax().item()

        result = {
            "condition": config.CLASS_NAMES[top_idx],
            "confidence": probs[top_idx].item(),
            "all_class_probs": {
                name: probs[i].item() for i, name in enumerate(config.CLASS_NAMES)
            },
            "image_vs_text_gate": output["gate"].item(),  
        }
        if return_attention:
            result["attn_weights"] = output["attn_weights"]
        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image_path> <symptom_text>")
        sys.exit(1)

    predictor = FusionPredictor(config.BEST_MODEL_PATH)
    result = predictor.predict(sys.argv[1], sys.argv[2])
    print(f"{result['condition']}  (confidence {result['confidence']:.2%}, "
          f"image_vs_text_gate={result['image_vs_text_gate']:.2f})")

import argparse

import torch
from torch.utils.data import DataLoader

import config
from cv_extractor import CVSpatialFeatureExtractor
from nlp_extractor import BioBERTTokenExtractor, MockBioBERTTokenExtractor
from fusion_model import SkinScanFusionModel, compute_loss
from dataset import PairedSkinScanDataset, collate_fn, generate_synthetic_manifest


def set_seed(seed=config.SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    
    backbone_params, head_params = [], []
    backbone_modules = (model.cv_extractor, model.nlp_extractor)
    for module in backbone_modules:
        for p in module.parameters():
            if p.requires_grad:
                backbone_params.append(p)

    backbone_param_ids = {id(p) for p in backbone_params}
    for p in model.parameters():
        if p.requires_grad and id(p) not in backbone_param_ids:
            head_params.append(p)

    param_groups = [{"params": head_params, "lr": config.LEARNING_RATE}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.BACKBONE_LEARNING_RATE})

    return torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, texts, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            output = model(images, texts)
            loss = compute_loss(output, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (output["fused_logits"].argmax(-1) == labels).sum().item()
            total_n += labels.size(0)

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true",
                         help="Generate + train on synthetic data as a pipeline smoke test.")
    parser.add_argument("--mock_text_backbone", action="store_true",
                         help="Use MockBioBERTTokenExtractor instead of real BioBERT "
                              "(no network/HF Hub access needed).")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()

    set_seed()

    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize((config.CV_IMAGE_SIZE, config.CV_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.synthetic:
        train_manifest = generate_synthetic_manifest("/content/drive/MyDrive/fusion/synthetic_train.csv", "/content/drive/MyDrive/fusion/synthetic_images")
        val_manifest = generate_synthetic_manifest("/content/drive/MyDrive/fusion/synthetic_val.csv", "/content/drive/MyDrive/fusion/synthetic_images", n_per_class=2)
    else:
        assert args.train_manifest and args.val_manifest, (
            "Pass --train_manifest and --val_manifest, or use --synthetic to smoke-test."
        )
        train_manifest, val_manifest = args.train_manifest, args.val_manifest

    train_ds = PairedSkinScanDataset(train_manifest, image_transform)
    val_ds = PairedSkinScanDataset(val_manifest, image_transform)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    cv_extractor = CVSpatialFeatureExtractor().to(config.DEVICE)
    if args.mock_text_backbone:
        nlp_extractor = MockBioBERTTokenExtractor().to(config.DEVICE)
    else:
        nlp_extractor = BioBERTTokenExtractor().to(config.DEVICE)

    model = SkinScanFusionModel(cv_extractor, nlp_extractor).to(config.DEVICE)
    optimizer = build_optimizer(model)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer=None)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  -> saved new best model ({best_val_acc:.3f}) to {config.BEST_MODEL_PATH}")

    print(f"Done. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()

def build_model(arch: str, num_classes: int, dropout: float = 0.0, pretrained: bool = True):
    arch = arch.lower()

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    elif arch in ["efficientnet", "efficientnet_b0"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unknown arch: {arch}")

    return m

def set_finetune_mode(model, finetune: str = "head"):
    """
    finetune="head": freeze backbone, train only classifier head
    finetune="all":  train everything
    """
    finetune = finetune.lower()
    if finetune == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif finetune == "head":
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze common head names
        for name, p in model.named_parameters():
            if any(k in name for k in ["fc", "classifier"]):
                p.requires_grad = True
    else:
        raise ValueError("finetune must be 'head' or 'all'")

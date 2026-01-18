def run_experiment(
    arch,
    finetune="head",
    lr=3e-4,
    weight_decay=1e-4,
    dropout=0.5,
    epochs=10,
    use_class_weights= False
):
    model = build_model(arch, num_classes=num_classes, dropout=dropout, pretrained=True)
    set_finetune_mode(model, finetune=finetune)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights if use_class_weights else None)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    best_wts = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    for ep in range(1, epochs+1):
        print(f"\n[{arch} | finetune={finetune}] Epoch {ep}/{epochs}")

        tr_loss, tr_acc, tsec = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  train: loss={tr_loss:.4f} acc={tr_acc:.4f} ({tsec:.1f}s)")
        print(f"  val:   loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)

    # test metrics
    y_true, y_pred, y_prob = predict_all(model, test_loader)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # weighted (by support) per-class accuracy
    supports = cm.sum(axis=1)
    per_class_acc = np.diag(cm) / np.maximum(supports, 1)
    weighted_acc = np.average(per_class_acc, weights=supports)

    out = {
        "arch": arch,
        "finetune": finetune,
        "model": model, # <--- IMPORTANT
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "epochs": epochs,
        "history": history,
        "test_acc": float(acc),
        "test_weighted_acc": float(weighted_acc),
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    return out

experiments = [
    {"arch":"resnet18", "finetune":"head"},
    {"arch":"resnet50", "finetune":"head"},
    {"arch":"efficientnet_b0", "finetune":"head"},
]

results = {}
for cfg in experiments:
    key = f"{cfg['arch']}_{cfg['finetune']}"
    results[key] = run_experiment(
        arch=cfg["arch"],
        finetune=cfg["finetune"],
        lr=3e-4,
        weight_decay=1e-4,
        dropout=0.5,
        epochs=10
    )

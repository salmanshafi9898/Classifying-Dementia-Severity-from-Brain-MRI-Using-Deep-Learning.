#Model Selection: Stage 1
#Build ResNet50 (head replaced)


import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_resnet50(num_classes: int):
    # Load ImageNet-pretrained ResNet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Replace final FC layer with 4-class head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

#Build train/eval loops

from tqdm import tqdm

def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    """
    One pass over a loader.
    If optimizer is None -> eval mode, no gradient updates.
    Returns average loss and accuracy.
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, leave=False):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def train_with_early_stopping(model, train_loader, val_loader,
                              criterion, optimizer,
                              device, num_epochs=30, patience=5):
    """
    Trains model with early stopping on validation accuracy.
    Returns best model (in-place) and history dict.
    """
    model.to(device)

    best_val_acc = 0.0
    best_state   = None
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device=device)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          optimizer=None, device=device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Early stopping on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = model.state_dict()
            epochs_no_improve = 0
            print("  ↳ New best model, saving state.")
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best val acc: {best_val_acc:.4f}")
    return model, history

#1. Build ResNet-50 with 4-class head

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_resnet50(num_classes: int = 4):
    """
    Create an ImageNet-pretrained ResNet-50 and replace the final FC layer
    with a new classifier for `num_classes`.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


#2. Generic train/eval helpers (used in both stages)

from tqdm import tqdm

def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    """
    One pass over a loader.
    If optimizer is None -> eval mode (no gradient updates).
    Returns average loss and accuracy.
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, leave=False):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def train_with_early_stopping(model, train_loader, val_loader,
                              criterion, optimizer,
                              device, num_epochs=30, patience=5):
    """
    Train the model with early stopping on validation accuracy.
    Returns model loaded with the best weights and a history dict.
    """
    model.to(device)

    best_val_acc = 0.0
    best_state   = None
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device=device)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          optimizer=None, device=device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Early stopping on val accuracy
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state   = model.state_dict()
            epochs_no_improve = 0
            print("  ↳ New best model, saving state.")
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best val acc: {best_val_acc:.4f}")
    return model, history


#3. Stage 1 – Train only the head (fc)

# ----- Stage 1: head-only training -----
model = build_resnet50(num_classes)
model.to(device)

# 1) Freeze ALL parameters except the final FC head
for name, param in model.named_parameters():
    if not name.startswith("fc."):   # conv1, bn1, layer1–4, etc. → frozen
        param.requires_grad = False

# Sanity check: only fc.* should be True
print("Stage 1 trainable params:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print("  ", n)

# 2) Optimizer on head only
trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer_stage1 = torch.optim.AdamW(
    trainable_params,
    lr=1e-3,          # slightly higher LR for small head
    weight_decay=1e-4
)

# 3) Train for a few epochs with early stopping on val accuracy

model, hist_stage1 = train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer_stage1,
    device=device,
    num_epochs=10,   # you can tweak
    patience=3       # stop if no val improvement for 3 epochs
)

print(model)

#4. Stage 2 – Unfreeze layer4 + fc and fine-tune

# ----- Stage 2: unfreeze layer4 + head -----

# 1) Freeze everything
for name, param in model.named_parameters():
    param.requires_grad = False

# 2) Unfreeze last ResNet block (layer4) and final FC layer
for name, param in model.named_parameters():
    if name.startswith("layer4") or name.startswith("fc"):
        param.requires_grad = True

print("Stage 2 trainable params:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print("  ", n)

# 3) Set different LRs for layer4 vs head
params_layer4 = [p for n, p in model.named_parameters()
                 if p.requires_grad and n.startswith("layer4")]
params_head   = [p for n, p in model.named_parameters()
                 if p.requires_grad and n.startswith("fc")]

optimizer_stage2 = torch.optim.AdamW(
    [
        {"params": params_layer4, "lr": 1e-5},  # smaller LR for pretrained convs
        {"params": params_head,   "lr": 1e-3},  # slightly larger for head
    ],
    weight_decay=1e-4
)

# Optional: scheduler that reduces LR when val accuracy plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_stage2, mode="max", factor=0.5, patience=2
)

def train_with_scheduler(model, train_loader, val_loader,
                         criterion, optimizer, scheduler,
                         device, num_epochs=30, patience=5):
    model.to(device)

    best_val_acc = 0.0
    best_state   = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device=device)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          optimizer=None, device=device)

        print(f"  train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Step scheduler on validation accuracy
        scheduler.step(val_acc)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state   = model.state_dict()
            epochs_no_improve = 0
            print("  ↳ New best model, saving state.")
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best val acc (stage 2): {best_val_acc:.4f}")
    return model

# 4) Run Stage 2 fine-tuning
model = train_with_scheduler(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer_stage2,
    scheduler,
    device=device,
    num_epochs=30,   # upper bound; early stopping will likely stop earlier
    patience=5
)

import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Acc')
    plt.plot(epochs, val_acc, 'r', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Usage:
# history = model.fit(...)
# plot_history(history)

#Grad-CAM Visualization

import numpy as np
import tensorflow as tf
import cv2

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map by the gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224)) # ResNet standard size

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

# Implementation for ResNet50
# heatmap = get_gradcam_heatmap(preprocessed_img, model, "conv5_block3_out")
# display_gradcam(original_img_path, heatmap)


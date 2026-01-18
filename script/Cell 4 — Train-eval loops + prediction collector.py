def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    t0 = time.time()

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return running_loss / total, running_correct / total, (time.time() - t0)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return running_loss / total, running_correct / total

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    all_y, all_pred, all_prob = [], [], []

    for x, y in tqdm(loader, desc="test", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_y.extend(y.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())

    return np.array(all_y), np.array(all_pred), np.array(all_prob)

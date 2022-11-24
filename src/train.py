import torch


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    loss = 0.0
    corrects = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        # compute loss
        pred = model(x)
        loss = loss_fn(pred, y)
        # backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss += loss.item() * x.size(0)
        corrects += torch.sum(pred.argmax(1) == y)

    epoch_loss = round((loss / len(dataloader.dataset)).item(), 5)
    epoch_acc = round((corrects / len(dataloader.dataset)).item(), 5)
    return epoch_loss, epoch_acc


def test(model, dataloader, loss_fn, device):
    n_batches = len(dataloader)
    model.eval()
    loss = 0.0
    corrects = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # compute loss
            pred = model(x)
            loss += loss_fn(pred, y).item()
            corrects += torch.sum(pred.argmax(1) == y)
    loss = round(loss / n_batches, 5)
    acc = round((corrects / len(dataloader.dataset)).item(), 5)
    return loss, acc

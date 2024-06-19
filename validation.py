import torch

def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient computation
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            if isinstance(targets, tuple):  # Handling CutMix targets
                targets1, targets2, lam = targets
                loss = lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)
            else:
                loss = criterion(outputs, targets)
            
            val_loss += loss.item() * data.size(0)  # Accumulate the validation loss
            pred = outputs.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    
    return val_loss, accuracy
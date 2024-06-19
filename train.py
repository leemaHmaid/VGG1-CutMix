from Model import VGG11
from cutmix import CutMixCollator
from validation import validate

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from CustomImageDataset import tra

model = VGG11(num_classes=1000).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduces LR by 0.1 every 10 epochs



num_epochs = 20
print_every = 1000  # Change this value to set how often to print

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    batch_loss = 0.0
    batch_count = 0

    for i, (data, targets) in enumerate(train_loader_cutmix , 1):
        data = data.to(device)
        
#         print(f"Batch {i} type of targets: {type(targets)}")  # Debugging: print type of targets
        if isinstance(targets, tuple):
            targets1, targets2, lam = targets
#             print(f"Batch {i} targets1 type: {type(targets1)}, targets2 type: {type(targets2)}")  # Debugging: print types of targets1 and targets2
            targets1, targets2 = targets1.to(device), targets2.to(device)
            targets = (targets1, targets2, lam)
        else:
            targets = targets.to(device)  # Ensure targets is a tensor and move to device
        
        optimizer.zero_grad()
        outputs = model(data)
        
        if isinstance(targets, tuple):  # Handling CutMix targets
            targets1, targets2, lam = targets
            loss = lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)
        else:
            # Convert each element in the list to a tensor and move to device
            if isinstance(targets, list):
                targets = [target.to(device) for target in targets]
                targets = torch.stack(targets)
            else:
                targets = targets.to(device)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        batch_loss += loss.item()
        batch_count += 1
        
        if i % print_every == 0:
            print(f'Batch {i}/{len(train_loader_cutmix )}, Loss: {batch_loss / print_every:.4f}')
            batch_loss = 0.0  # Reset batch loss for the next interval
    
    train_loss = running_loss / len(train_loader_cutmix.dataset)
    
    # Validate and print validation loss
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    # Save the model parameters after each epoch
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
    print(f'Model saved as model_epoch_{epoch+1}.pth')
    scheduler.step()

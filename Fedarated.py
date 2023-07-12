import os
import math
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import train_transform, val_transform
from dataset import get_train_val_test_Dataloaders
import torch.nn as nn

from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)

if BACKGROUND_AS_CLASS:
    NUM_CLASSES += 1

writer = SummaryWriter("runs")

model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
train_transforms = train_transform
val_transforms = val_transform

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=val_transforms)
print(len(train_dataloader))
print(len(val_dataloader))

def dice_loss(input, target):
    smooth = 1e-7
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice

def dice_coefficient(input, target):
    smooth = 1e-7
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

criterion = dice_loss
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

# Define the client training function
def client_train(model, train_dataloader, criterion, optimizer, client_name):
    model.train()
    train_loss = 0.0
    total_dice = 0.0
    num_samples = 0

    for data in train_dataloader:
        image, ground_truth = data['image'], data['label']
        image = image.reshape(1, 16, 240, 240, 160)
        ground_truth = ground_truth.squeeze(1)
        optimizer.zero_grad()
        target = model(image)
        target = target.squeeze(0)
        target = target.unsqueeze(1)
        target = target.repeat(1, 16, 1, 1, 1)
        ground_truth = ground_truth.repeat(target.size(0), 1, 1, 1, 1)
        loss = criterion(target, ground_truth)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Calculate accuracy (dice coefficient)
        dice = dice_coefficient(target, ground_truth)
        total_dice += dice * image.size(0)  # Accumulate the dice coefficient for batch-level accuracy
        num_samples += image.size(0)  # Increment the total number of samples

    train_accuracy = total_dice / num_samples
    print(f'Client: {client_name} \t Training Loss: {train_loss / len(train_dataloader)} \t Training Accuracy: {train_accuracy:.4f}')
    return model.state_dict(), train_loss / len(train_dataloader), train_accuracy

# Define a wrapper class for the global model
class GlobalModel(nn.Module):
    def __init__(self, model):
        super(GlobalModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

# Instantiate the global model with the wrapper class
global_model = GlobalModel(model)

# Define the server aggregation function
def aggregate_models(global_model, client_models):
    num_clients = len(client_models)

    # Initialize an empty model with the same architecture as the global model
    aggregated_model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)

    # Aggregate the model updates from all clients
    global_params = list(global_model.model.parameters())
    for client_model, client_name in client_models:
        client_params = list(client_model.parameters())
        for i in range(len(global_params)):
            global_params[i].data += client_params[i].data / num_clients

        # Calculate accuracy (dice coefficient)
        total_dice = 0.0
        num_samples = 0
        for data in train_dataloader:
            image, ground_truth = data['image'], data['label']
            image = image.reshape(1, 16, 240, 240, 160)
            ground_truth = ground_truth.squeeze(1)
            dice = dice_coefficient(client_model(image), ground_truth)
            total_dice += dice * image.size(0)  # Accumulate the dice coefficient for batch-level accuracy
            num_samples += image.size(0)  # Increment the total number of samples

        train_accuracy = total_dice / num_samples
        print(f'Client: {client_name} \t Training Accuracy: {train_accuracy:.4f}')

    # Load the aggregated model weights into the aggregated model
    aggregated_model.load_state_dict(global_model.model.state_dict())

    # Return the aggregated model and accuracy
    return aggregated_model, train_accuracy

# Set the number of clients and their names
num_clients = 2
client_names = ['Client1', 'Client2']

for epoch in range(TRAINING_EPOCH):
    print("initiating epoch", epoch + 1, "/", TRAINING_EPOCH, "-->")
    train_loss = 0.0
    train_accuracy = 0.0
    global_model.train()

    # Client training phase
    client_models = []
    for client_index in range(num_clients):
        for data in train_dataloader:
            image, ground_truth = data['image'], data['label']
            # Create a new client model
            client_model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
            client_model.load_state_dict(global_model.model.state_dict())  # Initialize the client model with the global model weights
            client_optimizer = Adam(params=client_model.parameters())
            client_criterion = dice_loss

            # Train the client model
            client_model_state_dict, client_loss, client_accuracy = client_train(client_model, train_dataloader, client_criterion, client_optimizer, client_names[client_index])

            # Save the client model
            client_model.load_state_dict(client_model_state_dict)
            client_models.append((client_model, client_names[client_index]))

            train_loss += client_loss
            train_accuracy += client_accuracy

    # Server aggregation phase
    aggregated_model, train_accuracy = aggregate_models(global_model, client_models)

    # Update the global model with the aggregated model
    global_model.load_state_dict(aggregated_model.state_dict(), strict=False)

    valid_loss = 0.0
    global_model.eval()
    for data in val_dataloader:
        image, ground_truth = data['image'], data['label']
        image = image.reshape(1, 16, 240, 240, 160)
        ground_truth = ground_truth.squeeze(1)
        optimizer.zero_grad()
        target = global_model(image)
        target = target.squeeze(0)
        target= target.unsqueeze(1)
        target = target.repeat(1, 16, 1, 1, 1)
        ground_truth = ground_truth.repeat(target.size(0), 1, 1, 1, 1)
        loss = criterion(target, ground_truth)
        valid_loss = loss.item()

    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)

    print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Training Accuracy: {train_accuracy:.4f} \t\t Validation Loss: {valid_loss}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        os.makedirs('checkpoints', exist_ok=True)  # Create the "checkpoints" directory if it doesn't exist
        torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss:.6f}.pth')

writer.flush()
writer.close()
import os
import math
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import train_transform, val_transform, train_transform_cuda, val_transform_cuda
from dataset import get_train_val_test_Dataloaders
import torch.nn as nn

from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)

if BACKGROUND_AS_CLASS:
    NUM_CLASSES += 1

writer = SummaryWriter("runs")

model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=val_transforms)
print("Total train batch :",len(train_dataloader))
print("Total val batch :",len(val_dataloader))

# def dice_loss(input, target):
#     smooth = 1e-7
#     input_flat = input.view(-1)
#     target_flat = target.view(-1)
#     intersection = (input_flat * target_flat).sum()
#     dice = (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
#     return 1 - dice

def dice_loss(input, target):
    smooth = 1e-7
    input_flat = input.view(input.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (input_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (input_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1 - dice.mean()

criterion = dice_loss
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

# Define the client training function
def client_train(model, train_dataloader, criterion, optimizer):
    model.train()
    train_loss = 0.0

    for data in train_dataloader:
        image, ground_truth = data['image'], data['label']
        # print(image.shape)

        # Reshape the image tensor
        batch_size = image.size(0)  # Get the batch size
        image = image.reshape(batch_size, 16, 240, 240, 160)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        model = model.to(device)
        
        # Perform other operations
        ground_truth = ground_truth.squeeze(1)
        optimizer.zero_grad()
        target = model(image)
        target = target.squeeze(0)
        target = target.unsqueeze(1)
        repeat_dims = (batch_size, 1, 1, 1, 1, 1)  # Adjust repeat dimensions
        target = target.repeat(*repeat_dims)

        # Ensure that target and ground_truth have the same size along dimension 0
        min_batch_size = min(target.size(0), ground_truth.size(0))
        target = target[:min_batch_size]
        ground_truth = ground_truth[:min_batch_size]

        # Ensure that target_flat and ground_truth_flat have the same number of elements along dimension 1
        min_num_elements = min(target.size(2), ground_truth.size(1))
        target_flat = target.view(target.size(0), -1)[:, :min_num_elements]
        ground_truth_flat = ground_truth.view(ground_truth.size(0), -1)[:, :min_num_elements]

        # print("Target shape : ",target.shape)
        # print("Ground truth shape : ",ground_truth.shape)

        # Continue with the rest of the code
        loss = dice_loss(target_flat, ground_truth_flat)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # print("**************** IT WORKS***************************")

    return model.state_dict(), train_loss / len(train_dataloader)

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
    for client_model in client_models:
        client_params = list(client_model.parameters())
        for i in range(len(global_params)):
            global_params[i].data += client_params[i].data / num_clients

    # Load the aggregated model weights into the aggregated model
    aggregated_model.load_state_dict(global_model.model.state_dict())

    # Return the aggregated model
    return aggregated_model

# Set the number of clients
num_clients = 2

for epoch in range(TRAINING_EPOCH):
    print("initiating epoch", epoch + 1, "/", TRAINING_EPOCH, "-->")
    train_loss = 0.0
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
            client_model_state_dict, client_loss = client_train(client_model, train_dataloader, client_criterion, client_optimizer)

            # Save the client model
            client_model.load_state_dict(client_model_state_dict)
            client_models.append(client_model)

            train_loss += client_loss

    # Server aggregation phase
    aggregated_model = aggregate_models(global_model, client_models)

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
        target = target.unsqueeze(1)
        target = target.repeat(1, 16, 1, 1, 1)
        ground_truth = ground_truth.repeat(target.size(0), 1, 1, 1, 1)
        loss = criterion(target, ground_truth)
        valid_loss = loss.item()

    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)

    print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss}')

    if min_valid_loss > valid_loss:
        # print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        # min_valid_loss = valid_loss
        # # Saving State Dict
        # torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        os.makedirs('checkpoints', exist_ok=True)  # Create the "checkpoints" directory if it doesn't exist
        torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')


writer.flush()
writer.close()




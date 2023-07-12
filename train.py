import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

writer = SummaryWriter("runs")

model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)
train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)
print(len(train_dataloader))
print(len(val_dataloader))

criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS*4))
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):
    print("initiating epoch",epoch+1,"/",TRAINING_EPOCH,"-->")
    train_loss = 0.0
    model.train()
    for data in train_dataloader:
        # image, ground_truth = data['image'], data['label']
        # print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        # print(data['image'].shape)
        # image = image.reshape( 1, 16, 240, 240, 160)
        # print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        # optimizer.zero_grad()
        # target = model(image)
        # print(target)
        # loss = criterion(target, ground_truth)
        # import torch
        image, ground_truth = data['image'], data['label']
        # print("Original image shape:", image.shape)
        # print("Original ground_truth shape:", ground_truth.shape)
        # image = image.reshape(1, 16, 240, 240, 160)
        image = image.reshape(1, 16, 240, 240, 160)
        # print("Reshaped image shape:", image.shape)
        ground_truth = ground_truth.squeeze(1)
        # print("Adjusted ground_truth shape:", ground_truth.shape)
        # ground_truth = ground_truth.unsqueeze(1)
        optimizer.zero_grad()
        target = model(image)
        # print("Target shape:", target.shape)
        target = target.squeeze(0)
        # print("Adjusted target shape:", target.shape)

        target = target.unsqueeze(1)  # Add a channel dimension to the target tensor
        target = target.repeat(1, 16, 1, 1, 1)  # Repeat the target tensor to match the number of channels in the ground truth tensor

        # 
        # ground_truth = ground_truth.expand(4, -1, -1, -1, -1)  # Adjust the batch size to 4

        # ground_truth = ground_truth.reshape(1, 16, 240, 240, 160)  # Reshape to 3D tensor

        ground_truth = ground_truth.repeat(target.size(0),1,1,1,1)
        loss = criterion(target, ground_truth)
        # loss = criterion(image, target)


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    valid_loss = 0.0
    model.eval()
    for data in val_dataloader:
        image, ground_truth = data['image'], data['label']
        image = image.reshape( 1, 16, 240, 240, 160)
        target = model(image)
        loss = criterion(target,ground_truth)
        valid_loss = loss.item()
        
    # writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    # writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
    
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss}')
    
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        # torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

# writer.flush()
# writer.close()


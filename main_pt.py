import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import argparse
import time
import numpy as np

from models.mobilenet_pt import MobileNetv1
from models.resnet_pt import ResNet8
from set_seed import set_random_seed

class Grayscale(object):
    def __call__(self, img):
        return transforms.functional.to_grayscale(img, num_output_channels=1)

class ContrastStretch(object):
    def __call__(self, img):
        img = transforms.functional.adjust_contrast(img, contrast_factor=1.08)
        return img


# Set Random Seed
set_random_seed(233)

mean_val = [0.4914, 0.4822, 0.4465]
std_val = [0.2470, 0.2435, 0.2616]
# Argument parser
parser = argparse.ArgumentParser(description='ECE361E Project - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=301, help='Number of epochs to train')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size

random_transform1 = transforms.RandomHorizontalFlip(p=0.5)
random_transform2 = transforms.Compose([transforms.Pad(padding=4),
                                            transforms.RandomCrop((32, 32))])



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_val,
                         std=std_val),
    transforms.RandomChoice([random_transform1, random_transform2]),

])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_val,
                         std=std_val),
])

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=train_transform)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=test_transform)

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = None
model_str = ""

params = {'in_channels': 3, 'out_channels': 10, 'activation': 'Default'}
model = ResNet8(params)
model_str = "ResNet8"



# Put the model on the GPU
device = torch.device('cuda')
model = model.to(device)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
total_training_time = 0
# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    start = time.time()
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Put the images and labels on the GPU
        images = images.to(device)
        labels = labels.to(device)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))

    end = time.time()
    per_epoch_training_time = end - start
    total_training_time += per_epoch_training_time
    train_loss_list.append(train_loss / (batch_idx + 1))
    train_acc_list.append(100. * train_correct / train_total)

    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Put the images and labels on the GPU
            images = images.to(device)
            labels = labels.to(device)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), 100. * test_correct / test_total))

    test_loss_list.append(test_loss / (batch_idx + 1))
    test_acc_list.append(100. * test_correct / test_total)

    # Save the PyTorch model in .pt format
    path = f'ckpt/{model_str}.pt'
    torch.save(model.state_dict(), path)


with open(f'{model_str}.csv', 'w+') as csv_file:
    csv_file.write(f"Epoch,Train acc,Train loss,Test acc,Test loss\n")
    for epoch in range(num_epochs):
        csv_file.write(f"{epoch},{train_acc_list[epoch]},{train_loss_list[epoch]},{test_acc_list[epoch]},{test_loss_list[epoch]}\n")

total_param_list = [p for p in model.parameters()]
trainable_param_list = [p for p in model.parameters() if p.requires_grad]
num_trainable_params = 0
for param in trainable_param_list:
    num_trainable_params += np.prod(param.cpu().data.numpy().shape)

print('Overall training accuracy at the end of %d epochs : %.4f' % (num_epochs, train_acc_list[-1]) + '%')
print('Overall test accuracy at the end of %d epochs : %.4f' % (num_epochs, test_acc_list[-1]) + '%')
print('Total time for training : %.4f seconds' % total_training_time)
print('Total number of trainable parameters : %d' % num_trainable_params)

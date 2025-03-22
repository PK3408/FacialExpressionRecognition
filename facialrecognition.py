import numpy as np
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as func
import torchvision.transforms as transforms
import torch.optim as optim
import cv2 as cv
import pandas as pd
import os

#define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the dataset
# Dataframes consisting of their respective collections of happy, sad, and surprised faces
# These images must first be mass converted to 2D arrays using OpenCV

trainingSet = pd.DataFrame()
testingSet = pd.DataFrame()

def convert_images_to_arrays(image_folder, label):
    image_list = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')): # Check if the file is an image
            img_path = os.path.join(image_folder, filename)
            img = cv.imread(img_path)
            if img is not None:
                image_list.append(np.array([img,label],dtype=object))
            else:
                print(f"Error reading image: {filename}")
    return np.array(image_list)

# Example usage:
train_image_folder_happy = 'archiveDataset/train/happy'
train_image_folder_sad = 'archiveDataset/train/sad'
train_image_folder_surprise = 'archiveDataset/train/surprise'

test_image_folder_happy = 'archiveDataset/test/happy'
test_image_folder_sad = 'archiveDataset/test/sad'
test_image_folder_surprise = 'archiveDataset/test/surprise'

train_image_array_happy = convert_images_to_arrays(train_image_folder_happy,0)
print(train_image_array_happy.shape)
train_image_array_sad = convert_images_to_arrays(train_image_folder_happy,1)
print(train_image_array_sad.shape)
train_image_array_surprise = convert_images_to_arrays(train_image_folder_happy,2)
print(train_image_array_surprise.shape)

test_image_array_happy = convert_images_to_arrays(test_image_folder_happy,0)
print(test_image_array_happy.shape)
test_image_array_sad = convert_images_to_arrays(test_image_folder_happy,1)
print(test_image_array_sad.shape)
test_image_array_surprise = convert_images_to_arrays(test_image_folder_happy,2)
print(test_image_array_surprise.shape)

train_image_array = np.concatenate([train_image_array_happy,train_image_array_sad,train_image_array_surprise])
print(train_image_array.shape)

test_image_array = np.concatenate([test_image_array_happy,test_image_array_sad,test_image_array_surprise])
print(test_image_array.shape)

if train_image_array.size > 0:
    print(f"Successfully converted {len(train_image_array)} {len(test_image_array)} images to arrays.")
    # Further processing with image_arrays (e.g., saving to a file)
else:
     print("No images were converted.")

#load the dataset
batch_size = 4

trainloader = torch.utils.data.DataLoader(train_image_array, batch_size=batch_size,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(test_image_array, batch_size=batch_size,
                                         shuffle=False)

print("successfully loaded dataset")

#CNN
net = Net()

#use cross entropy loss and set parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#train the dataset
#has a problem when running more than one epoch, look into it later
for epoch in range(0):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#test the model
correct = 0
total = 0
with torch.no_grad():
    for data in enumerate(testloader):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network: {100 * correct // total} %')


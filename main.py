
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from PIL import Image
import torch.optim as optim
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import torch

import numpy as np
from models import CNN_model
from models import VGG16_model

from image_dataset import ImageDataSet
from keras.utils import to_categorical


# Define paths to the train, test, and validation folders
# train_path = '/FloodNet-Supervised_v1.0/train'
# test_path = '/FloodNet-Supervised_v1.0/test'
# val_path = '/FloodNet-Supervised_v1.0/val'

#train_path = '/Users/jovinbains/Desktop/Computer Vision Project/Comp4102_project/FloodNet-Supervised_v1-3.0/train'
#test_path = '/Users/jovinbains/Desktop/Computer Vision Project/Comp4102_project/FloodNet-Supervised_v1-3.0/test'
# val_path = '/Users/jovinbains/Desktop/Computer Vision Project/Comp4102_project/FloodNet-Supervised_v1-3.0/val'

train_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-3-2.0/train'
test_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-3-2.0/test'
#val_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-3.0/val'


class_names = {
    'Background': 0,
    'Building-flooded': 1,
    'Building-non-flooded': 2,
    'Road-flooded': 3,
    'Road-non-flooded': 4,
    'Water': 5,
    'Tree': 6,
    'Vehicle': 7,
    'Pool': 8,
    'Grass': 9
}

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

train_dataset = ImageDataSet(root_dir=train_path, class_number=class_names['Pool'], transform=transform)
test_dataset = ImageDataSet(root_dir=test_path, class_number=class_names['Pool'],transform=transform)
# val_dataset = CustomDataSet(root_dir=val_path , transform=transform)


# Define the CNN Class
model = CNN_model()
# Define the ResNet Class
input_shape = (32, 32, 3)
vgg_model = VGG16_model(input_shape)
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_images = np.array([np.array(img) for img, _ in train_dataset])
train_labels = np.array([np.array(label) for _, label in train_dataset])
train_images = train_images.transpose((0, 2, 3, 1))

test_images = np.array([np.array(img) for img, _ in test_dataset])
test_labels = np.array([np.array(label) for _, label in test_dataset])
test_images = test_images.transpose((0, 2, 3, 1))


model.fit(
    train_images,
    train_labels,
    epochs=100,
    validation_data=(test_images, test_labels)
)

train_labels_one_hot = to_categorical(train_labels, num_classes=10)
test_labels_one_hot = to_categorical(test_labels, num_classes=10)


vgg_model.fit(
    train_images,
    train_labels_one_hot,
    epochs = 100,
    validation_data=(test_images,test_labels_one_hot)
)

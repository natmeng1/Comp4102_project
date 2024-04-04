import gdown
from googledriver import download_folder
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from PIL import Image



#URL = 'https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD?usp=share_link'
#download_folder(URL)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.originals = ImageFolder(root=root + '/org-img', transform=transform)
        self.labels = ImageFolder(root=root + '/label-img', transform=transform)

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        original_img, _ = self.originals[idx]
        label_img, _ = self.labels[idx]

        return original_img, label_img

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

# Define paths to the train, test, and validation folders
train_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1.0/train'
test_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1.0/test'
val_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1.0/val'

train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=test_path , transform=transform)
val_dataset = ImageFolder(root=val_path , transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

classes = ('Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded', 
           'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass')

# Define the CNN Class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Step 1: Create the first convolutional layer (conv1)
        #   - Input channels: 3
        #   - Output channels: 6
        #   - Kernel size: 5
        self.conv1 = nn.Conv2d(3,6,5)
        #self.conv1 = nn.Conv2d(3,6,3, padding='same')
        # Step 2: Create max pooling layer (pool)
        #   - Kernel size: 2
        #   - Stride: 2
        self.pool=nn.MaxPool2d(2,2)
        # Step 3: Create the second convolutional layer (conv2)
        #   - Input channels: 6
        #   - Output channels: 16
        #   - Kernel size: 5
        self.conv2 = nn.Conv2d(6,16,5)
        #self.conv2 = nn.Conv2d(6,16,3, padding='same')
        
        # Step 4: Create the first fully connected layer (fc1)
        #   - Input features: 16 * 5 * 5
        #   - Output features: 120
        self.fc1= nn.Linear(16 * 5 * 5, 120)
        #self.fc1= nn.Linear(16 * 5 * 5, 120)
        # Step 5: Create the second fully connected layer (fc2)
        #   - Input features: 120
        #   - Output features: 120
        self.fc2 = nn.Linear(120,120)
        # Step 6: Create the third fully connected layer (fc3)
        #   - Input features: 120
        #   - Output features: 10 (number of classes in CIFAR-10)
        self.fc3=nn.Linear(120,10)
    def forward(self, x):
        # Step 7: Define the forward pass
        #   - Apply conv1, followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.sigmoid(self.conv1(x)))
        
        #   - Apply conv2, followed by ReLU activation and max pooling

        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.sigmoid(self.conv2(x)))
        #   - Reshape the tensor for the fully connected layers
        
        #x = x.view(-1, 2304)
        x = x.view(x.size(0), -1)
        #   - Apply fc1, followed by ReLU activation
        #x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc1(x))
        
        #   - Apply fc2, followed by ReLU activation
        #x = F.sigmoid(self.fc2(x))
        x = F.relu(self.fc2(x))

        #   - Apply fc3 (output layer)
        x = self.fc3(x)
       
        #   - Return output
        return x
print(Net())
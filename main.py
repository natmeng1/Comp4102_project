
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from PIL import Image
import torch.optim as optim
from PIL import UnidentifiedImageError


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
train_path = '~/Downloads/FloodNet-Supervised_v1-2.0/train'
test_path = '~/Downloads/FloodNet-Supervised_v1-2.0/test'
val_path = '~/Downloads/FloodNet-Supervised_v1-2.0/val'

#train_path = 'FloodNet-Supervised_v1.0/train'
#test_path = 'FloodNet-Supervised_v1.0/test'
#val_path = 'FloodNet-Supervised_v1.0/val'

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


net=Net()

CUDA=torch.cuda.is_available()
if CUDA:
  net=net.cuda()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
accuracy_values=[]
epoch_number=[]

for epoch in range(3):  # loop over the dataset multiple times. Here 10 means 10 epochs
    running_loss = 0.0
    
    for i, (inputs,labels) in enumerate(train_loader, 0):
       
        try:

        # get the inputs; data is a list of [inputs, labels]
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

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
                print('[epoch%d, itr%5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        except UnidentifiedImageError:
            print("Image Corrupt")

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if CUDA:
              images = images.cuda()
              labels = labels.cuda()
            else:
              images = images.cpu()
              labels =labels.cpu()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if CUDA:
              correct += (predicted.cpu()==labels.cpu()).sum().item()
            else:
              correct += (predicted==labels).sum().item()

        TestAccuracy = 100 * correct / total;
        epoch_number += [epoch+1]
        accuracy_values += [TestAccuracy]
        print('Epoch=%d Test Accuracy=%.3f' %
                  (epoch + 1, TestAccuracy))

print('Finished Training')


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for images, labels in test_loader:
        if CUDA:
          images =images.cuda()
          labels =labels.cuda()
        else:
          images =images.cpu()
          labels =labels.cpu()

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    if(class_total[i] != 0):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    else:
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 0 ))



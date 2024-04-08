
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




 #URL = 'https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD?usp=share_link'
#download_folder(URL)


class CustomDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.org_dir=os.path.join(root_dir,'org-img')
        self.label_dir=os.path.join(root_dir,'label-img')
        self.org_images = os.listdir(self.org_dir)       
        self.label_images = os.listdir(self.label_dir)


    def __len__(self):
        return min(len(self.org_images), len(self.label_images))

    def __getitem__(self, idx):
        original_img_name = self.org_images[idx]
        label_img_name = self.label_images[idx]

        org_img_path = os.path.join(self.org_dir, original_img_name)
        label_img_path = os.path.join(self.label_dir,label_img_name)

        original_image = Image.open(org_img_path).convert("RGB")
        label_image = Image.open(label_img_path).convert("RGB")

        if self.transform:
            original_image = self.transform(original_image)
            label_image = self.transform(label_image)
        return original_image, label_image
    
# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

# Define paths to the train, test, and validation folders
train_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-2.0/train'
test_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-2.0/test'
val_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-2.0/val'

#train_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-3.0/train'
#test_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-3.0/test'
#val_path = '/Users/natmengistu/Downloads/FloodNet-Supervised_v1-3.0/val'



train_dataset = CustomDataSet(root_dir=train_path, transform=transform)
test_dataset = CustomDataSet(root_dir=test_path , transform=transform)
val_dataset = CustomDataSet(root_dir=val_path , transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,drop_last=True)
classes = ('Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded', 
           'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass')
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
        #self.fc1= nn.Linear(490000, 120)
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
        #print("x: ",len(x))
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
            #print("inputs: ",inputs.size())
            #should be:4,3,32,32
            #is:4,3,32,32
            print("labels b4: ", labels.size())
            print("out: ", outputs.size())
            _, predicted_labels = torch.max(outputs, dim=1)
            print("labels after: ", predicted_labels.size())
            #should be:4,
            #is:4,

            outputs = net(inputs)
            #print(outputs)
            #print("outputs: ", outputs.size())
             #should be:4,10
            #is:4,10
        
            loss = criterion(outputs, predicted_labels)
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
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
           # print("AFTER CPMP label: ", labels.size())
            #print("AFTER CMP PREDICTED: ", predicted.size())
            if CUDA:
             # print("Here")
              correct += (predicted.cpu()==labels.cpu()).sum().item()
            else:
             # print("Else")
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
        for i in range(32):
            label = labels[i]
            _, label_flat = torch.max(label, 1)
            print("LABELL: ", label)
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    if(class_total[i] != 0):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    else:
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 0 ))



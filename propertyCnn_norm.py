import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# DATA
num_classes = 10
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2) 

# MODEL PARAMS
L = 10**-2
num_epochs = 5
train = False


# MODEL - adapted from https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
class PropertyNet(nn.Module):
    def __init__(self):
        super(PropertyNet, self).__init__()
        # TODO should this be 3, 32, 3 bc 3-channel images?
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3), \
                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 3), \
                nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(8, 16, 3), \
                nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, 3), \
                nn.ReLU(inplace=True))

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400, num_classes)
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4, self.pool, self.fc1])
        self.pool_output = torch.randn(0, 400)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x = self.conv3(x)
        x = self.pool(self.conv4(x))
        x = x.view(-1, 400)

        # save activations into environ variable
        self.pool_output = torch.cat([self.pool_output, x], 0)
        x = self.fc1(x)
        return x
property_net = PropertyNet()

# TRAINING
if train:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(property_net.parameters(), lr=L)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = property_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if not i % 200:
                print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 200))
                running_loss = 0.0
    torch.save(property_net.state_dict(), "models/propCnn_unnorm")
property_net.load_state_dict( torch.load('models/propCnn_unnorm') )

property_net.pool_output = torch.randn(0, 400)
print('Finished training')
property_net.eval()

# SAVE TESTING ACTIVATIONS
correct = 0
total = 0
all_outputs = np.array([])
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = property_net(images)
        all_outputs = np.vstack([all_outputs, outputs]) if all_outputs.size else outputs
        __, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

np.save('output_activations', all_outputs)
print(property_net.pool_output.shape, correct/total)
torch.save(property_net.pool_output, 'test_unnorm_activations.dat')

# SAVE TRAINING ACTIVATIONS
property_net.pool_output = torch.randn(0, 400)
all_outputs = np.array([])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2)
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs = property_net(images)

print(property_net.pool_output.shape)
torch.save(property_net.pool_output, 'train_unnorm_activations.dat')

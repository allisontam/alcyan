import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# DATA
num_classes = 10
modes = ['L', 'RGB'] 
transform = transforms.Compose([transforms.Grayscale(), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.5], [0.5])])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(list(range(32))))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2) 
# testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(list(range(32))))


transform_rgb = transforms.Compose([transforms.ToTensor()])

trainset_rgb = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
trainloader_rgb = torch.utils.data.DataLoader(trainset_rgb, batch_size=16, shuffle=False, num_workers=2)

testset_rgb = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
testloader_rgb = torch.utils.data.DataLoader(testset_rgb, batch_size=16, shuffle=False, num_workers=2)



# MODEL PARAMS
L = 10**-2
num_epochs = 5


# MODEL - adapted from https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
class PropertyNet(nn.Module):	
    def __init__(self, mode):
        super(PropertyNet, self).__init__()	# mode = RGB -> 3-channel color images; mode = L -> 1-channel grayscale images
        self.mode = 3 if mode == 'RGB' else 1
        self.conv1 = nn.Conv2d(1 * self.mode, 8 * self.mode, 3)	# TODO should this be 3, 32, 3 bc 3-channel images?
        self.conv2 = nn.Conv2d(8 * self.mode, 8 * self.mode, 3)
        self.conv3 = nn.Conv2d(8 * self.mode, 16 * self.mode, 3)
        self.conv4 = nn.Conv2d(16 * self.mode, 16 * self.mode, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400 * self.mode, num_classes)
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4, self.pool, self.fc1])
        self.pool_output = torch.randn(0, 400 * self.mode)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 400 * self.mode)
        self.pool_output = torch.cat([self.pool_output, x], 0)
#         torch.save(self.pool_output, 'pool_output.dat') 
        x = self.fc1(x)
        return x
property_net = PropertyNet(mode='RGB')
final_trainloader = trainloader_rgb if property_net.mode == 3 else trainloader
final_testloader = testloader_rgb if property_net.mode == 3 else testloader


# TRAINING
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(property_net.parameters(), lr=L)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(final_trainloader, 0):
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

property_net.pool_output = torch.randn(0, 400)
print('Finished training')


# TESTING
correct = 0
total = 0
all_outputs = np.array([])
with torch.no_grad():
    for data in final_testloader:
        images, labels = data
        outputs = property_net(images)
#         print(property_net.layers[4].output.shape)
        all_outputs = np.vstack([all_outputs, outputs]) if all_outputs.size else outputs        
        __, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i += 1

# np.save('output_activations', all_outputs)
print(property_net.pool_output.shape)
torch.save(property_net.pool_output, 'all_pool_output_3.dat' if property_net.mode == 'RGB' else 'all_pool_output.dat')

print('Accuracy of network on 10000 test images: %d %%' % (100 * correct / total))



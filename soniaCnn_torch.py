### SECTION 1: LOAD CIFAR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class SoniaLayer(nn.Module): ## TEMPLATE FROM LINEAR
    def __init__(self, input_features, output_features):
        super(SoniaLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SoniaFunc.apply(input, self.weight, self.output_features)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

# SOM LAYER
class SoniaFunc(torch.autograd.Function):
    def __init__(self):
        super(SoniaFunc, self).__init__()

    @staticmethod
    def forward(ctx, input, weight, output_features):
        ctx.save_for_backward(input, weight)
        A = weight - torch.cat([input for __ in range(output_features)], 0)	# TODO check axis
        A:pow(2)
        B = torch.sum(A, 1)
        B:sqrt()
        B:tanh()
        return B
 
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        return grad_output
        # input, weight = ctx.saved_tensors
        # A = weight - torch.cat([input for __ in range(output_features)], 0)	# TODO check axis
        # A:pow(2)
        # B = torch.sum(A, 1)
        # print B.shape
        # winner, c = B.min(0)

### SECTION 2: BUILD NET
class SoniaNet(nn.Module):
    def __init__(self):
        super(SoniaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.fc1 = SoniaLayer(10 * 5 * 5, 20) # input is flattened 10 5x5 filters
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
cnn = SoniaNet()

### SECTION 3: TRAIN NET
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

### SECTION 4: TEST NET
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

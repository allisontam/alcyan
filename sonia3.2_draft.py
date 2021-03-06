### SECTION 0: HYPERPARAMETERS
num_epochs = 1
hidden_init = 10
max_nodes = 10
sl = 250 # stimulation level
lr = 0.001
dl = 100 # distance level - 0 < dl < 1

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

class SoniaLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(SoniaLayer, self).__init__()
        self.input_features = input_features

        # cannot modify through gradient unless it's a float
        self.hidden_num = nn.Parameter(torch.ones(1)*hidden_init)

        # Initialize weights
        weight = torch.Tensor(output_features, input_features)
        weight.data.uniform_(-0.1, 0.1)
        weight[hidden_init:,:] = 0
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SoniaFunc.apply(input, self.weight, self.hidden_num)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, num_hidden_nodes={}'.format(
            self.in_features, self.out_features, self.hidden_num
        )

# SOM LAYER
class SoniaFunc(torch.autograd.Function):
    def __init__(self):
        super(SoniaFunc, self).__init__()

    @staticmethod
    def forward(ctx, input, weight, hidden_num):
        ctx.save_for_backward(input,
                              weight,
                              hidden_num)
        A = weight - torch.cat([input for __ in range(weight.size(0))], 0) # TODO: check axis
        A:pow(2)
        B = torch.sum(A, 1)
        # taking this line out to make the derivative more tangible
        # B:sqrt()
        B:tanh()
        B.unsqueeze_(0)
        return B
 
    @staticmethod
    def backward(ctx, grad_output):
        '''
        Three things you want to mask:
        1. forward activations from masked nodes should be 0
            Prevents FC layer weights from changing
            Handled in SoniaNet initialization
        2. weight matrices to masked node should be 0
            Allows you do initialize correctly when you unmask
        3. gradient that you give to conv layers should be 0
            Prevents conv layer weights from changing
        '''
        input, weight, hidden_num = ctx.saved_tensors
        grad_weight = torch.zeros(weight.shape)
        grad_hidden_num = torch.zeros(1)

        A = weight - torch.cat([input for __ in range(weight.size(0))], 0)
        B = torch.sum(torch.pow(A,2), 1)
        winner, c = B[:int(hidden_num)].min(0)

        # need to be able to adjust lr bc gradient calcs and step happens separately
        # implicitly handles (2)
        if winner < sl:
            grad_weight[c, :] = -winner/lr*0.1
        # else:
            # if hidden_num < max_nodes:
                # grad_hidden_num -= 1/lr
                # grad_weight[int(hidden_num),:] = -input/lr

        # based on math
        chain_grad = -A
        B.unsqueeze_(1)
        tanh_grad = torch.cat([B for __ in range(input.size(1))], 1)
        tanh_grad = 1-tanh_grad.pow(2)
        do_dx = chain_grad*tanh_grad
        do_dx[int(hidden_num):, :] = 0 # don't want to affect masked convolution weights (3)
        grad_input = grad_output.mm(do_dx) # element multiplication
        return grad_input, grad_weight, grad_hidden_num


### SECTION 2: BUILD NET
class SoniaNet(nn.Module):
    def __init__(self):
        super(SoniaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.sonia = SoniaLayer(10 * 5 * 5, max_nodes) # input is flattened 10 5x5 filters
        self.fc2 = nn.Linear(max_nodes, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = self.sonia(x)
        x = self.fc2(x)
        return x
cnn = SoniaNet()

### SECTION 3: TRAIN NET
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=lr)
#, momentum=0.9) potentially cannot use momentum because we need to control grad
for epoch in range(num_epochs):  # loop over the dataset multiple times

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
            print('number of hidden units', cnn.sonia.hidden_num)
            running_loss = 0.0

print('Finished Training')

### SECTION 5: ADD NODES + RETRAIN (3.3)
# 3.3: Create new mutation-based hidden units
# TODO sort weight vectors based on paper sort notes
# EDIT HERE
# Calculate distances
# for j in range(weight.size(0) - 1):
    # a = weight[j, :]
    # b = weight[j + 1, :]
    # d_ab = torch.sum(torch.pow(a - b, 2))
    # if d_ab > dl: # if we haven't exceeded the number of allowed mutations
        # for j in range(weight.size(0)):
            # # if we get to an all-zero row
            # if (weight[j, :] == torch.zeros(1, weight.size(1))).byte().all().item():
                # weight[j, :] = (torch.rand((1, weight.size(1))) * (a - b)) + b
                # break
# print(weight)

### SECTION 6: TEST NET
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

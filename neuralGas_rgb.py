import kohonen, torchvision, torch, pickle
from collections import defaultdict, Counter
import torchvision.transforms as transforms
import numpy as np

### HYPERPARAMETERS
max_nodes = 300
params = kohonen.kohonen.GrowingGasParameters(dimension=1200, shape=(max_nodes,), \
        growth_interval=20, max_connection_age=20, rapid_expand=10)
num_epochs = 1

### LOAD DATA
num_classes = 10
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# creating training matrix
inputs = torch.load('rgbData/all_pool_training_3.dat').detach().numpy() # train_actiations
truth = [int(data[1]) for data in trainloader]
# for data in trainloader:
    # truth.append(int(data[1]))
# 50,000 x 401 array. 1st col is truth
data = np.hstack([np.array(truth).reshape(len(truth),1), inputs])
np.random.shuffle(data)
trainset = np.copy(data[:,1:])

### TRAIN MODEL
# gas = kohonen.kohonen.ImmuneGas(params) # initialization
gas = kohonen.kohonen.GrowingGas(params) # initialization
for epoch in range(num_epochs):
    for cue in trainset:
        gas.learn(cue)
    np.random.shuffle(trainset)
# pickle.dump(gas, open('gas-30.pkl','wb'))

### TRANSFORM MODEL INTO CLASSIFIER
# with open('gas.pkl','rb') as f:
    # gas = pickle.load(f)
cluster_rep = defaultdict(list)
for entry in data:
    truth = int(entry[0])
    cue = entry[1:]
    cluster_rep[gas.winner(cue)].append(truth)

for k in cluster_rep.keys():
    cluster_rep[k] = np.argmax(np.bincount(np.array(cluster_rep[k])))
print('class represenation amongst nodes:', Counter(cluster_rep.values()))

# load testing set
inputs = torch.load('rgbData/all_pool_output_3.dat').detach().numpy() #testing_activations
truth = [int(data[1]) for data in testloader]
# 10,000 x 401 array. 1st col is truth
data = np.hstack([np.array(truth).reshape(len(truth),1), inputs])
print(data.shape,'should be 10000x401')
# testset = np.copy(data[:,1:])

### TEST MODEl
correct, tot = 0,0
wrong = []
for entry in data:
    truth = int(entry[0])
    winner = gas.winner(entry[1:])
    if cluster_rep[winner] == truth:
        correct += 1
    else:
        wrong.append(int(truth))
    tot += 1
print('accuracy:', correct/tot)
print('class representation amongs incorrectly classified:', Counter(wrong))

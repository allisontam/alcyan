import kohonen, torchvision, torch, pickle
from collections import defaultdict, Counter
import torchvision.transforms as transforms
import numpy as np

### TRY NOT TO USE THIS VERSION. USE THE OTHER ONE. THIS JUST HAS THE IMAGES.

### HYPERPARAMETERS
max_nodes = 30
train_size = 9000
params = kohonen.kohonen.GrowingGasParameters(dimension=400, shape=(30,), growth_interval=5)
num_epochs = 5
train_run = True

### LOAD DATA
num_classes = 10
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
inputs = torch.load('all_pool_output.dat').numpy()

truth = []
for data in testloader:
    truth.append(int(data[1]))

# 10,000 x 401 array. 1st col is truth
data = np.hstack([np.array(truth).reshape(len(truth),1), inputs])
np.random.shuffle(data)
trainset = np.copy(data[:train_size,1:])
testset = data[train_size:]


### TRAIN MODEL
if train_run:
    gas = kohonen.kohonen.GrowingGas(params) # initialization
    for epoch in range(num_epochs):
        for cue in trainset:
            gas.learn(cue)
        np.random.shuffle(trainset)

    pickle.dump(gas, open('gas-30.pkl','wb'))

### TRANSFORM MODEL INTO CLASSIFIER
if not train_run:
    with open('gas-30.pkl','rb') as f:
        gas = pickle.load(f)

img = gas.neuron_heatmap()
img.save('test.png')

cluster_rep = defaultdict(list)
for entry in data[:train_size]:
    truth = int(entry[0])
    cue = entry[1:]
    cluster_rep[gas.winner(cue)].append(truth)

for k in cluster_rep.keys():
    cluster_rep[k] = np.argmax(np.bincount(np.array(cluster_rep[k])))
print('class represenation amongst nodes:', Counter(cluster_rep.values()))

correct, tot = 0,0
wrong = []
for entry in testset:
    truth = int(entry[0])
    winner = gas.winner(entry[1:])
    if cluster_rep[winner] == truth:
        correct += 1
    else:
        wrong.append(int(truth))
    tot += 1
print('accuracy:', correct/tot)
print('class representation amongs incorrectly classified:', Counter(wrong))

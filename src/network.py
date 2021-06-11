import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

net=CNNNetwork().to(device)

file_path = 'C:\BIAI-Projekt\src\data\punkty.txt'
with open(file_path) as file:
    data = []
    for line in file.readlines():
        b = line.strip().split('\t')
        point = (float(b[0]), float(b[1]))
        data.append(point)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

tensors=[]
for i in data:
    tensor = torch.tensor([i[0],i[1]])
    tensors.append(tensor)

print('Start Training')

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, point in enumerate(tensors):
        # get the inputs; data is a list of [inputs, labels]
        x=point[0].to(device)
        y=point[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
        
PATH = './cnn.pth'
torch.save(net.state_dict(), PATH)


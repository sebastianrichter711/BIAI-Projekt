import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import seaborn as sns
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.preprocessing import MinMaxScaler

file_path = 'C:\BIAI-Projekt\src\data\punkty.txt'
with open(file_path) as file:
    data = []
    for line in file.readlines():
        b = line.strip().split('\t')
        point = (float(b[0]), float(b[1]))
        data.append(point)

all_data = []
for i in data:
    got_point = i
    got_y = got_point[1]
    all_data.append(got_y)

print(all_data)

test_data_size=51

old_train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

print(len(old_train_data))
print(len(test_data))
print(test_data)

train_data = np.array(old_train_data)
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

print(train_data_normalized[:5])
print(train_data_normalized[-5:])

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

train_window = 51

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

print(train_inout_seq[:5])

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

epochs = 360

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 51

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

print(test_inputs[fut_pred:])

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)

x = np.arange(300, 351, 1)
print(x)

x_new = np.arange(0,351,1)
print(x_new)

plt.title('Prediction of a point movement')
plt.ylabel('Y')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(x_new, all_data)
plt.plot(x,actual_predictions)
plt.show()
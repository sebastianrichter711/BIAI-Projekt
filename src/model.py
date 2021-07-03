import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

TEST_DATA_SIZE = 10 # declare how many points we're going to predict
scaler = MinMaxScaler(feature_range=(-1, 1))

#create a class representing neural network
#parameters of a class:
    # - input size -  Corresponds to the number of features in the input (for us we deliver a y coordinate of one point)
    # - hidden_layer_size - number of neurons in the network
    # - output size - the number of items in the output (here we get predicted y coordinate of one point given in the input)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size) # these variables are used to create the LSTM and linear layer
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), # this variable contains the previous hidden and cell state
            torch.zeros(1,1,self.hidden_layer_size))

    #The predicted y coordinate of analized point is stored in the last item of the predictions 
    #list, which is returned to the calling function
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class Simulation_model:
    def __init__(self):
        self.model = LSTM()
        self.loss_function = nn.MSELoss() #define a cross entropy loss function and the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.all_data = []
        self.all_data_size = 0
        self.points = 50 #at first we train a neural network for first 50 points
        self.predicted_points = []
        self.epochs = self.points + 13
        self.input_x_values = []
        self.output_x_values = []
        
    #function will accept the raw input data and will return a list of tuples
    #In each tuple, the first element will contain list of N items corresponding to the y coordinates of N points
    @staticmethod
    def create_inout_sequences(input_data):
        inout_seq = []
        L = len(input_data)
        for i in range(L-TEST_DATA_SIZE):
            train_seq = input_data[i:i+TEST_DATA_SIZE]
            train_label = input_data[i+TEST_DATA_SIZE:i+TEST_DATA_SIZE+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq


     #parse input data
    def read_data(self, path: str):
        #open file
        with open(path) as file:
            self.all_data = []
            for line in file.readlines():
                line = line.strip()
                #if line doesn't match following regex, the input file is invalid
                if re.match(r"-?[0-9]+.[0-9]+\t-?[0-9]+.[0-9]+", line) == None:
                    return False
                vec = line.split('\t')
                point = float(vec[1])
                self.all_data.append(point)
            
            self.all_data_size = len(self.all_data)
            self.input_x_values = np.arange(0, self.all_data_size, 1).tolist()
            self.output_x_values = np.arange(51, self.all_data_size, 1).tolist()
            return True

    def simulate(self):
        self.points = 50
        while self.points < self.all_data_size - 1:  #while we achieve the end of list of analized points
            old_training_data = self.all_data[:self.points] #get first N(50,60,...) points from list of all points to train
            training_data = np.array(old_training_data) 
            #Normalization of data to range <-1,1>
            normalized_training_data = scaler.fit_transform(training_data.reshape(-1, 1))
            #convert our dataset into tensors
            normalized_training_data = torch.FloatTensor(normalized_training_data).view(-1)
            training_inout_seq = self.create_inout_sequences(normalized_training_data)

            self.epochs = self.points + 13 # training a network for N+13 epochs
            for _ in range(self.epochs):
                for seq, labels in training_inout_seq:
                    self.optimizer.zero_grad()
                    self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                    torch.zeros(1, 1, self.model.hidden_layer_size))

                    y_pred = self.model(seq)

                    single_loss = self.loss_function(y_pred, labels)
                    single_loss.backward()
                    self.optimizer.step()

            #get last N samples from a normalized training dataset to make predictions
            test_inputs = normalized_training_data[-TEST_DATA_SIZE:].tolist()

            #make predictions
            #in for loop items from test_inputs list will be used to make predictions about the first item from the test set
            self.model.eval()

            for _ in range(TEST_DATA_SIZE): #for each point to predict
                seq = torch.FloatTensor(test_inputs[-TEST_DATA_SIZE:])
                with torch.no_grad():
                    self.model.hidden = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                    torch.zeros(1, 1, self.model.hidden_layer_size))
                    test_inputs.append(self.model(seq).item()) #The predict value will then be appended to the test_inputs list

            actual_predictions = scaler.inverse_transform(np.array(test_inputs[TEST_DATA_SIZE:] ).reshape(-1, 1)) #convert the normalized predicted values into actual predicted values

            #add predicted points in this iteration to the list of all predicted points
            for point in actual_predictions:
                self.predicted_points.append(point)
            
            self.points += TEST_DATA_SIZE  #add next N points to train a network

    #method returning whether any data was added to list
    def is_data_loaded(self):
        return self.all_data_size != 0

            



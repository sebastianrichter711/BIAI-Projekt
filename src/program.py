import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    filename = input("Enter filename to analyze: ")

    #Reading data from a text file with given name.
    file_path = f'{os.getcwd()}\data\{filename}'
    with open(file_path) as file:
        data = [] #create a list of points
        for line in file.readlines():
            b = line.strip().split('\t')
            point = (float(b[0]), float(b[1]))
            data.append(point)

    #Selecting y coordinate of all analized points
    all_data = [] # create a list for this aim
    for i in data:
        got_point = i
        got_y = got_point[1] #get y coordinate
        all_data.append(got_y)

    print(all_data) # show all y coordinates of all points
    number_all_data = len(all_data) # get number of all points and print it
    print(number_all_data)

    predicted_points = [] # create a list storing points showing anticipated movement of point
    test_data_size=10 # declare how many points we're going to predict
    number_of_points=50 #at first we train a neural network for first 50 points

    while number_of_points < number_all_data-1: #while we achieve the end of list of analized points
        old_train_data = all_data[:number_of_points] #get first N(50,60,...) points from list of all points to train 
        #test_data = all_data[-test_data_size:]

        print(len(old_train_data))
        #print(len(test_data))
        #print(test_data)

        train_data = np.array(old_train_data) #create a numpy array based on training data to train a network
        #Normalization of data to range <-1,1>
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

        #print the first 5 and last 5 records of our normalized train data
        print(train_data_normalized[:5])
        print(train_data_normalized[-5:])

        #convert our dataset into tensors
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
        
        #convert our training data into sequences and corresponding labels
        #length of train window depends on number of points to predict
        train_window = test_data_size

        #function will accept the raw input data and will return a list of tuples
        #In each tuple, the first element will contain list of N items corresponding to the y coordinates of N points
        def create_inout_sequences(input_data, tw):
            inout_seq = []
            L = len(input_data)
            for i in range(L-tw):
                train_seq = input_data[i:i+tw]
                train_label = input_data[i+tw:i+tw+1]
                inout_seq.append((train_seq ,train_label))
            return inout_seq

        #Execute the following script to create sequences and corresponding labels for training
        train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

        # print the first 5 items of the above list
        print(train_inout_seq[:5])

        #create a class representing neural network
        #parameters of a class:
        # - input size -  Corresponds to the number of features in the input (for us we deliver a y coordinate of one point)
        # - hidden_layer_size - number of neurons in the network
        # - output size - the number of items in the output (here we get predicted y coordinate of one point given in the input)
        class LSTM(nn.Module):
            def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
                super().__init__()
                self.hidden_layer_size = hidden_layer_size

                self.lstm = nn.LSTM(input_size, hidden_layer_size) # these variables are used to create the LSTM and linear layers

                self.linear = nn.Linear(hidden_layer_size, output_size) 

                self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), # this variable contains the previous hidden and cell state
                                torch.zeros(1,1,self.hidden_layer_size))

            #The predicted y coordinate of analized point is stored in the last item of the predictions 
            #list, which is returned to the calling function
            def forward(self, input_seq):
                lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
                predictions = self.linear(lstm_out.view(len(input_seq), -1))
                return predictions[-1]

        model = LSTM() # create an object representing neural network
        loss_function = nn.MSELoss() #define a cross entropy loss function and the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(model) # show data about our network

        # training a network for N+13 epochs
        epochs = number_of_points + 13

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()
            #after every 25 epochs the loss will be printed
            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        #define a variable representing number of points to predict
        fut_pred = test_data_size

        #get last N samples from a normalized training dataset to make predictions
        test_inputs = train_data_normalized[-train_window:].tolist()
        print(test_inputs)

        #make predictions
        #in for loop items from test_inputs list will be used to make predictions about the first item from the test set
        model.eval()

        for i in range(fut_pred): #for each point to predict
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                test_inputs.append(model(seq).item()) #The predict value will then be appended to the test_inputs list

        print(test_inputs[fut_pred:]) #print got results

        #convert the normalized predicted values into actual predicted values
        actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
        print(actual_predictions)

        #add predicted points in this iteration to the list of all predicted points
        for point in actual_predictions:
            predicted_points.append(point)

        #add next N points to train a network
        number_of_points += test_data_size

    #create x ranges for all and predicted data (to plot)
    x = np.arange(51, number_all_data, 1)
    print(x)

    x_new = np.arange(0, number_all_data, 1)
    print(x_new)

    #Show a plot with all essential data
    plt.title('Prediction of a point movement')
    plt.ylabel('Y')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(x_new, all_data)
    plt.plot(x, predicted_points)
    plt.show()
#import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("weatherHistory.csv")
print(data)
#to know #.of rows&columns
data.shape
#give us info about data
data.info()
#calculate the percentage of null values in columns
print(round(100*(data.isnull().sum()/len(data.index)),2))  #OR-----> data.isnull().sum()
data.head()
#to show a statistical table about this data(#describetive statistics)
data.describe()
#in the stat table the col"loud cover"=0 in all data
#show us the values in this col and counts the #of each value
data['Loud Cover'].value_counts()
#copy data
CopyData=data.copy()
CopyData.isnull().sum()
#try to impute null with max occured value
#show us the values in this col and counts the #of each value
print(CopyData['Precip Type'].value_counts())
#replace all null values in a column Precip Type whitch will be 'rain'#rain occured more than snow
CopyData.loc[CopyData['Precip Type'].isnull(),'Precip Type']='rain'
#calculate the percentage of null values in columns
CopyData.isnull().sum()
#OR----->print(round(100*(CopyData.isnull().sum()/len(CopyData.index)),2))
#removing Null values &zero
CopyData=CopyData.drop(['Loud Cover'],axis=1)
#Extract features from 'Formatted Date' column
CopyData['Formatted Date']=pd.to_datetime(data['Formatted Date'],utc=True)

#convert this to year & month & day
CopyData['year']=CopyData['Formatted Date'].dt.year
CopyData['month']=CopyData['Formatted Date'].dt.month
CopyData['day']=CopyData['Formatted Date'].dt.day
CopyData.head()
#drop formated date
CopyData.drop('Formatted Date',axis=1,inplace=True)
CopyData.head(3)
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 11, 10
CopyData.hist()
Pressure_mean = CopyData['Pressure (millibars)'].mean()


def pressure(x):
    if x == 0:
        return x + Pressure_mean
    else:
        return x


CopyData["Pressure (millibars)"] = CopyData.apply(lambda row: pressure(row["Pressure (millibars)"]), axis=1)

rcParams['figure.figsize'] = 5, 3
CopyData['Pressure (millibars)'].hist()
#to show the data type of each col
CopyData.dtypes
#want to delete this col because this is an object
#give some info about this col
CopyData["Daily Summary"].unique()
#drop this col
CopyData.drop('Daily Summary',axis=1,inplace=True)
#to show the data type of each col
CopyData.dtypes
#deal with summary col because this is an object
len(CopyData["Summary"].unique())

#give some info about this col
CopyData["Summary"].unique()
#replacing the column values to fit into majority of the values.
#to treat with min # of values
CopyData['Summary']=CopyData['Summary'].replace(['Breezy and Overcast','Breezy and Mostly Cloudy','Breezy and Partly Cloudy','Breezy and Foggy'],'Breezy')
CopyData['Summary']=CopyData['Summary'].replace(['Dry and Partly Cloudy','Dry and Mostly Cloudy','Windy and Dry','Breezy and Dry'],'Dry')
CopyData['Summary']=CopyData['Summary'].replace(['Windy and Partly Cloudy','Windy and Overcast','Windy and Mostly Cloudy','Windy and Foggy','Dangerously Windy and Partly Cloudy'],'Windy')
CopyData['Summary']=CopyData['Summary'].replace(['Light Rain','Drizzle'],'Rain')
CopyData['Summary']=CopyData['Summary'].replace(['Humid and Mostly Cloudy','Humid and Partly Cloudy','Humid and Overcast'],'Humid')

#convert each value to numeric value
summaryOfDictionary={'Partly Cloudy':0,
            'Mostly Cloudy':1,
            'Overcast':2,
            'Clear':3,
            'Foggy':4,
            'Breezy':5,
            'Dry':6,
            'Windy':7,
            'Rain':8,
            'Humid':9,}

#to convert the summary col to int data type
CopyData=CopyData.replace({"Summary": summaryOfDictionary})
CopyData.head()
CopyData.dtypes
#treate with 'Precip Type' col
CopyData['Precip Type']=CopyData['Precip Type'].replace(["rain"],1)
CopyData['Precip Type']=CopyData['Precip Type'].replace(["snow"],0)
CopyData.dtypes
# to split the data set into trainig dataset &testing dataset
Training_data = CopyData.drop(['Humidity'], axis=1)
test_output = CopyData[["Humidity"]]
#Multilayer Artificial Neural Network (Mulilayear-ANN)
#The backpropagation algorithm

from random import seed
from random import random
from math import exp

#Step(1): Initialize the network "First, Hidden, and Output Layers"
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    	#--
    hidden_layer = [];
    neuron= {'weights':[]}
    for i in range(n_hidden):
        for j in range(n_inputs+1):
            neuron['weights'].append(random());
        hidden_layer.append(neuron)
    #--
    #hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights of hidden to output':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
#-------
seed(1) #method is used to initialize the random number generator
network = initialize_network(12, 1, 1)
for layer in network:
    print(layer)

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1] #Remember the last index in weights list "Bais"
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i] #bais in the activation and I sum over it
        print('inputs',inputs)
        print('activation',activation)
        input('In activation Cont')
    return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))#ٍSigmoed

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        print('layer: ',layer)
        for neuron in layer:
            print('neuron',neuron['weights'])
            activation = activate(neuron['weights'], inputs)
            print('all_activation', activation)
            neuron['output'] = transfer(activation)
            print('output after transfer: ',neuron['output'])
            new_inputs.append(neuron['output'])
            inputs = new_inputs
            print("update inputs: ",inputs)
    return inputs
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output) #Derivative of sigmoid function

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        print('The layer num: ',i,' = ',layer, 'error= ',errors)
        #---error calculations
        if i != len(network)-1: #not the last item
            for j in range(len(layer)): #j represent neuron in layer[i]
                error = 0.0
                print('j neuron: ',j)
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                print('j neuron error: ',error)
                errors.append(error)
                print('j neuron error: ',error, 'Updated errors: ', errors)
        else: #the last item
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
                print('At the last layer for each neuron: errors= ', errors)
        #---Delta calculation
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
            print('For neuron= ',neuron,' calculated delta: ',neuron['delta'])
        input('Cont. to new layer')

#number of epochs is the number of times that the entire training dataset is shown to the network during training
numberOfEpochs = 10
epochSize = int((len(CopyData)) / numberOfEpochs)

import numpy as np

numberOfEpochs = 50
epochSize = int((len(Training_data)) / numberOfEpochs)

# Randomly set the weights’ values.
weights = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1),
           np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
bias = np.random.uniform(0, 1)
learningRate = 0.0000005

epochStart = 0
epochEnd = epochSize

epoch = Training_data.iloc[epochStart:epochEnd]
desiredOutput = epoch.iloc[:, 2]
# epoch = epoch.drop('Humidity', axis='columns')

# Return a new array S of shape:epochSize and type:float
S = np.empty(epochSize, float)

# Return a new array error of shape:epochSize and type:float.
error = np.empty(epochSize, float)
deltaWeights = np.empty((epochSize, 7), float)
deltaBias = np.empty(epochSize, float)

for k in range(numberOfEpochs):
    for i in range(epochSize):
        #  S[i] = np.dot(epoch.iloc[i], weights) + bias
        error[i] = S[i] - desiredOutput.iloc[i]

        # deltaWeights[i] = np.dot(epoch.iloc[i], error[i] * learningRate)
        deltaBias[i] = learningRate * error[i]

    # update the weights
    weights = weights - np.mean(deltaWeights)
    # update the biase
    bias = bias - np.mean(deltaBias)

    epochStart = epochStart + epochSize
    epochEnd = epochEnd + epochSize
    epoch = Training_data.iloc[epochStart:epochEnd]
    desiredOutput = epoch.iloc[:, 2]
# epoch = epoch.drop('Humidity', axis='columns')


epochSize = int((len(test_output)) / numberOfEpochs)
epochStart = 0
epochEnd = epochSize

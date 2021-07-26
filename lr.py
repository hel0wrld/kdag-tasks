import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class LogRegression:
    '''
        Class to make an object for Logistic Regression
        ----------
        Functions:

        initialize:
                Initializes the weight, X and Y matrix needed for calculations

                Arguments:
                        x: The dataset containing the features/properties of the classes
                        y: A list of 0/1s denoting which class an entry belongs to
                Returns:
                        <None>\n
        SigmoidFunction:
                Finds the sigmoid of the dot product of weight and X matrix

                Arguments:
                        w: The weight matrix needed for calculations
                        x: The dataset containing the features of the data
                Returns:
                        Scalar (numpy.float64)\n
        Gradient:
                Calculates the gradient for applying the gradient descent algorithm\n
        Update:
                Updates the weights according to the learing rate and number of iterations provided\n
        Classifier:
                To predict the class from which the data belongs to      
    '''
    def initialize(self, x, y):
        self.temp = np.ones((x.shape[0], 1))
        self.x = np.concatenate((self.temp,x), axis=1)
        self.wt = np.zeros(self.x.shape[1])
        self.y = y
    
    #Sigmoid function to convert values to the range [0,1]
    def SigmoidFunction(self, x, w):
        z = np.dot(x, w)
        val = 1/(1+np.exp(-z)) #the expression np.exp() can raise warnings which are suppressed by the lines 3-4
        return val

    #Gradient calculator for updating the weights in each iteration
    def Gradient(self, x, h, y):
        m = y.shape[0]
        val = np.dot(x.T, (h-y))/m
        return val

    #Fitting the model, i.e., updating the weights according to the losses and learning rate
    def Update(self, rate, iterations):
        for i in range(iterations):
            sigma = self.SigmoidFunction(self.x, self.wt)
            self.wt-=rate*self.Gradient(self.x, sigma, self.y)

    #Classifier function, which classifies the data into the 2 binary states
    def Classifier(self, x, lim):
        output = []
        x = np.concatenate((self.temp,x), axis=1)
        final = self.SigmoidFunction(x, self.wt)
        final = final>=lim
        for i in range(final.shape[0]):
            if final[i]==True:
                output.append(1)
            else:
                output.append(0)
        return output

data = ['ds1_test.csv', 'ds1_train.csv', 'ds2_test.csv', 'ds2_train.csv'] 
#Iterating through the list of all the datasets
lr = [0.1, 0.161, 0.9, 0.8]
#List of learning rates for each of the dataset
epoch = [9000, 5000, 100, 1000]
#List of epoch/number of iterations for each of the dataset
for i, sample in enumerate(data):
    print('Making a model for',sample)
    dataset = pd.read_csv(sample).to_numpy()            #Using pandas to read the csv file
    x = dataset[:,0:2]                                  #Slicing the csv to get our dataset                                  
    y = dataset[:,2]
    model = LogRegression()                             #Making an object of the LogRegression class
    model.initialize(x, y)                              #Initializing it with the datasets
    model.Update(lr[i], epoch[i])                       #Updating the weights
    prediction = model.Classifier(x, 0.5)               #Finally classifying the data 
    print('Model created succesfully')
    #Printing the accuracy of our model
    print(f"Accuracy for {sample} = {sum(prediction==y)/y.shape[0]}")


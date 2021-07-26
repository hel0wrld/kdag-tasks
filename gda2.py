import numpy as np
import pandas as pd

class GDA:
    '''
        Class to make an object for Gaussian Discriminant Analysis
        ------------
        Functions:

        CalcParameters:   

                Calculates the basic parameters for a model such as phi, mean and the covariance matrix

                Arguments: 
                        x: The dataset containing the features/properties of the classes
                        y: A list of 0/1s denoting which class an entry belongs to
                Returns:
                        <None>

        Classifier:

                Classifies the data into 2 classes based upon the features learnt by finding the parameters

                Arguments: 
                        x: The dataset containing the features/properties of the classes
                        threshold: The limit above which the data is labelled as that of a particular class 
                Returns:
                        list of 0/1s predicting which class the data belongs to
    '''
    def CalcParameters(self, x, y):
        ''' 
            Calculates the value of phi, mean and the covariance matrix:\n

                phi = (Number of indices where y=1)/Number of entries\n
                mean_0/1 = (Sum of x's where y=0/1)/Number of indices where y=0/1\n
                covariance matrix = According to the formula\n
        '''
        #Finding the number of samples and features
        n_features = x.shape[1]
        self.classes = [0,1]
        self.y=y.astype(int)
        self.x=x
        
        #Initializing the phi, mean and covariance variables according to the size 
        self.phi = np.mean(y==1)
        self.mean = np.zeros((2, n_features))
        self.covariance = 0
        
        for i in range(2):
            self.mean[i] = np.mean(x[y==i], axis=0)                 #Finding the mean of this particular class 
        m = len(self.y)
        for i in range(m):
            x_minus_mu = self.x[i]-self.mean[self.y[i]]
            x_minus_mu = x_minus_mu.reshape(*(x_minus_mu.shape),1)  #Reshaping to avoid dimension errors while manipulating
            self.covariance+=np.matmul(x_minus_mu, x_minus_mu.T)
        self.covariance/=m

    def Classifier(self, x, threshold):
        '''
            Classifies the input dataset into classes based on the features learnt
        '''
        output = []
        pi = np.pi
        n = len(self.mean[0])
        dr = (2*pi)**(n/2)*np.sqrt(np.linalg.det(self.covariance))
        for x in self.x:
            #Calculating p(x|y=0)
            x_mu0 = x - self.mean[0]
            x_mu0 = x_mu0.reshape(*(x_mu0.shape), 1)
            px_y0 = np.exp(-0.5*np.matmul(x_mu0.T, np.matmul(np.linalg.inv(self.covariance), x_mu0)))/dr
            px_y0 = np.squeeze(px_y0)

            #Calculating p(x|y=1)
            x_mu1 = x - self.mean[1]
            x_mu1 = x_mu1.reshape(*(x_mu1.shape), 1)
            px_y1 = np.exp(-0.5*np.matmul(x_mu1.T, np.matmul(np.linalg.inv(self.covariance), x_mu1)))/dr
            px_y1 = np.squeeze(px_y1)

            #Calculating p(x) = p(x|y=1)*p(y=1) + p(x|y=0)*p(y=0)
            Dr = px_y1*self.phi+px_y0*(1-self.phi)
            #Calculating p(y=1|x) = p(x|y=1)*p(y=1) / p(x)
            p_y1 = px_y1*self.phi/Dr
            #Labelling it into the 2 classes based on the threshold
            if p_y1>threshold:
                output.append(1)
            else:
                output.append(0)

        return output


data = ['ds1_test.csv', 'ds1_train.csv', 'ds2_test.csv', 'ds2_train.csv'] 
#Iterating through the list of all the datasets
for sample in data:
    print('Making a model for',sample)
    dataset = pd.read_csv(sample).to_numpy()            #Using pandas to read the csv file
    x = dataset[:,:2]                                   #Slicing the csv to get our dataset
    y = dataset[:, 2]
    model = GDA()                                       #Making an object of the GDA class
    model.CalcParameters(x, y)                          #Learning the features/Finding the required parameters
    result = model.Classifier(x,0.5)                    #Classifying the input data based on the threshold 
    print('Model created successfully')
    print('Accuracy for',sample,'=',(sum(y==result)/y.shape[0]))    #Printing the accuracy
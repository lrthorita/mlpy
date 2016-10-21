# -*- coding: utf-8 -*-
# **************************************************************************** #
# Autor:    LUIZ RICARDO TAKESHI HORITA
# E-mail:   lrthorita@usp.br
# **************************************************************************** #
#    Mestrado em Engenharia Elétrica e Computação     -     No.USP: 9917093
# **************************************************************************** #
""" ================  Multilayer Perceptron - Backpropagation ==================

============================================================================ """
#from math import *
import numpy as np


class mlpy:
    ## ********************* __init__ Method ******************************** ##
    
    def __init__(self,structure):
        """
        structure: It is a tuple variable that has the number of neurons in each
        layer. This variable must have at least 3 elements (one for input layer,
        one for hidden layer, and one for output layer). For example:
            
                            structure = (5, 3, 2, 1)
        
        This variable has 1 input layer with 5 neurons, 2 hidden layers with 3 
        neurons and 2 neurons each, and 1 output layer with just 1 neuron.
        
        Given the number of neurons in each layer of the ANN it is created a 
        dictionary containing matrices corresponding to the weights between a 
        layer and the  previous one. This function returns this dictionary.
        Each weights matrix is labeled as "layer" plus the number corresponding
        to its position. It starts from the first hidden layer, and its matrix 
        corresponds to all the weights between the current hidden layer and the 
        input layer. So, the last layer corresponds to the weights between the 
        output layer and the last hidden layer.
        For example: given the values on the example above...
            
                        architecture = {('layer1': matrix(3,5+1)),
                                        ('layer2': matrix(2,3+1)),
                                        ('layer3': matrix(1,2+1))}
        """
        ## THE CLASS VARIABLES:
        
        # The mlp structure contain the number of neurons in each layer.
        self.structure = structure
        # Create a dictionary containing all the weights respective to each layer
        self.architecture = {}
        # Create a dictionary containing all the outputs respective to each layer
        self.outputs = {} 
        # 
        self.MSE = 0
        # Number of iterations on the training step
        self.iterations = 0
        
        
        ## THE ARCHITECTURE INITIALIZATION:
             
        # Number of layers
        nLayers = len(structure)
        if(nLayers < 3):
            print "ERROR: mlpStructure must be a tuple variable with at least 3 elements!"
            return 
        else:
            for i in range(1,nLayers):
                layerName = "layer" + str(i)
                self.architecture[layerName] = np.random.rand(structure[i],structure[i-1]+1)*2-1
        
    ## ************************* Functions ********************************** ##
    def initWeights(self):
        # Number of layers
        nLayers = len(self.structure)
        for i in range(1,nLayers):
            layerName = "layer" + str(i)
            self.architecture[layerName] = np.random.rand(self.structure[i],self.structure[i-1]+1)*2-1
    
    
    def setupDataset(self,dataset):
        """
        In this class, it is considered to get the dataset as matrix form. Each 
        line of this matrix correspond to an example, and the input vector of 
        the NN is given by the transposition of a given line of this matrix. But 
        the 'nOutput' last elements of each line (or the 'nOutput' last columns of
        this matrix) represent the class of the examples.
        Then, this function separates this matrix in two matrices. It will 
        return a matrix containing just the input vectors (including the bias as
        the first element of each vector) of each example, and a matrix 
        containing the respective classes.
        """
        # The number of output neurons
        nOutput = self.structure[len(self.structure)-1]
        
        # Dictionary variable containing the matrices of examples and their labels
        datas = {}
        
        # Input datas and labels separation
        inputDataset = dataset[:,0:(np.size(dataset,1) - nOutput)]
        datas['labels'] = dataset[:,(np.size(dataset,1) - nOutput):np.size(dataset,1)]
        
        # Addin BIAS to the input dataset
        datas['examples'] = np.c_[np.ones((np.size(dataset,0),1)),inputDataset]
        
        return datas
    
    
    def sigmoid(self,net):
        output = 1/(1+np.exp(-net))
        return output
    
    
    def forward(self, inputVector):
        # The number of layes that give an output
        nLayers = len(self.architecture)
        
        for i in range(1,nLayers+1):
            currLayer = "layer" + str(i)
            
            # Compute the somatories of inputs and weights, plus bias
            net = np.dot(self.architecture[currLayer],inputVector)
            
            # Compute the activation function (given as a sigmoid function)
            output = self.sigmoid(net)
            self.outputs[currLayer] = output
            
            # Use the result as the input vector of the next layer
            inputVector = np.r_[np.ones((1,1)),output]
    
        
    def backpropagation(self, dataset, etta=0.1, alpha=0.0, limit=1e10, 
                        tolerance=1e-4):
        """
        Inputs:
            - dataset;
            - etta = learning rate (usually between [0.01 , 0.1]);
            - alpha = momentum factor (usually between [0.1 , 0.9]);
            - limit = limit of iterations on training step;
            - tolerance = error tolerance that defines when to stop the training.
        """
        nLayers = len(self.architecture) # The number of layes that give an output
        N = np.size(dataset,0) # Number of examples in the dataset
        datas = self.setupDataset(dataset)
            
        # Initially, there is no momentum.
        prevArchitecture = self.architecture.copy()
        
        """
        If the number of iterations is not defined, then the training will 
        happen according to the tolerance condition.
        """
        # Error and iteration initialization to get in the training loop
        self.MSE = 2 * tolerance
        self.iterations = 0
        while((self.MSE > tolerance) and (self.iterations < limit)):
            self.MSE = 0 # Start with no error
            
            # Shuffle the examples order in the dataset (for better training)
            index = np.arange(N)
            np.random.shuffle(index)
            
            for r in range(N):
                p = index[r]
                x_p = np.transpose(datas['examples'][p:p+1,:])
                y_p = np.transpose(datas['labels'][p:p+1,:])
                
                self.forward(x_p)
                outputLayer = 'layer' + str(nLayers)
                outputY = self.outputs[outputLayer]
                error = y_p - outputY
                sqErrorSum = np.sum(error**2)
                
                # TRAINING
                # ========
                if(sqErrorSum != 0): # Update only if there is error
                    self.MSE = self.MSE + sqErrorSum
                    
                    # DELTAS AND MOMENTUMS
                    # ====================
                    delta = {}
                    momentum = {}
                    
                    # Output Delta (OUTPUT LAYER)
                    delta[outputLayer] = (error) * outputY * (1 - outputY)
                    # Output Momentum (OUTPUT LAYER)
                    momentum[outputLayer] = self.architecture[outputLayer] - prevArchitecture[outputLayer]
                    momentum[outputLayer] = alpha * momentum[outputLayer]
                    
                    # Deltas and momentums of the HIDDEN LAYERS
                    for i in range(1,nLayers):
                        currLayer = nLayers - i
                        currLabel = 'layer' + str(currLayer)
                        nextLabel = 'layer' + str(currLayer+1)
                        
                        # The delta computation
                        nextDelta = np.transpose(delta[nextLabel])
                        nextWeights = self.architecture[nextLabel][:,1:np.size(self.architecture[nextLabel],1)]
                        delta[currLabel] = np.dot(nextDelta, nextWeights)
                        derivativeY = self.outputs[currLabel]*(1-self.outputs[currLabel])
                        delta[currLabel] = derivativeY*np.transpose(delta[currLabel])
                        
                        # The momentum computation
                        momentum[currLabel] = self.architecture[currLabel] - prevArchitecture[currLabel]
                        momentum[currLabel] = alpha * momentum[currLabel]
                    
                    # Save current architecture as prevArchitecture for next iteration
                    prevArchitecture = self.architecture.copy()
                    
                    # UPDATE ARCHITECTURE
                    # ===================
                    # Update the first hidden layer
                    deltaPart = etta * np.dot(delta['layer1'],np.transpose(x_p))
                    self.architecture['layer1'] = self.architecture['layer1'] + momentum['layer1'] + deltaPart
                    
                    for r in range(nLayers-1):
                        currLayer = nLayers - r
                        currLabel = 'layer' + str(currLayer)
                        prevLabel = 'layer' + str(currLayer-1)
                        
                        # Update architecture
                        inputVector = np.r_[np.ones((1,1)),self.outputs[prevLabel]]
                        deltaPart = etta * np.dot(delta[currLabel],np.transpose(inputVector))
                        self.architecture[currLabel] = self.architecture[currLabel] + momentum[currLabel] + deltaPart
                    
            # Compute the mean squared error (MSE)
            self.MSE = self.MSE/N
            
            # Increment training iteration
            self.iterations = self.iterations + 1
            
            # Show the MSE
            print "Iteration = " + str(self.iterations) + "; Total Squared Error = " + str(self.MSE)
        
        
            
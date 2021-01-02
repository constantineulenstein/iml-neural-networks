import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pickle
import numpy as np
import pandas as pd
#import wandb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,MinMaxScaler
from pickle import dump, load
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('error', RuntimeWarning)


from matplotlib import pyplot as plt

     

class Regressor():

    def __init__(self, x, nb_epoch = 100,lr = 0.002,wd = 0, batch_size = 56):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
    
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.
            
            - lr {float} -- Learning rate for the optimiser
            - wd {int} -- Weight decay

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Innitialize parameters of the Network
        X, _ = self._preprocessor(x, training = True)
        self.X = X
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.batch_size = batch_size
        self.running_loss = 0
        self.folds = 10
        self.lr = lr
        self.early_stopping = False
        self.weight_decay = wd
        # set tot TRUE to print the progress and evaluation metrics
        self.verbose = True
        
        # Define general structure of Network which can be later adjusted
        self.network = Network(input_dim = self.input_size,
                               output_dim = self.output_size)
        
        # Define the Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), 
                                    lr=self.lr, betas=(0.9, 0.999),
                                    eps=1e-08, weight_decay=self.weight_decay,
                                    amsgrad=False)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # --------------------------------------------------------------------
        # SORT THE DATA
        column_names = ['longitude', 'latitude', 'housing_median_age', 
                        'total_rooms','total_bedrooms', 'population', 
                        'households', 'median_income','ocean_proximity'] 
        # numerical features
        numeric_features = ['longitude', 'latitude', 'housing_median_age', 
                            'total_rooms','total_bedrooms', 'population', 
                            'households', 'median_income']
        # get the numerical features
        features = x[column_names]
        
        # --------------------------------------------------------------------
        # HANDLE CATEGORICAL FEATURES
        # Get dummies to transform categorical to Numerical
        features = pd.get_dummies(features)
        
        # Make sure the features are present in the dataset
        if 'ocean_proximity_ISLAND' not in features.columns.values:
            features['ocean_proximity_ISLAND'] = 0
        elif 'ocean_proximity_NEAR BAY' not in features.columns.values:
            features['ocean_proximity_NEAR BAY'] = 0
        
        #Drop one column to avoid multicolineariy: 'ocean_proximity_NEAR OCEAN'
        features = features[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
       'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY']]
        
        outputs = y
    
        # --------------------------------------------------------------------
        # PRE PROCESSING
        if(training):
            
            #Imput median value to missing values and rescale 
            numeric_transformer = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaler', StandardScaler())
            ])
            
            #Transform data to numeric, pass through others
            ct = ColumnTransformer(
                transformers = [
                    ('num', numeric_transformer, numeric_features),
                ],remainder = 'passthrough')
            
            #Processed data transformed
            df_processed = ct.fit_transform(X = features)
            
        # --------------------------------------------------------------------
        # SAVE MODEL 
            #Save  the Transfomer in a pkl file
            dump(ct, open("x_transformer.pkl","wb"))
            
            #Transform y -> is probably not necessary
            if y is not None:
                y_scaler = MinMaxScaler()
                outputs = y_scaler.fit_transform(outputs)
                dump(y_scaler, open("y_transformer.pkl","wb"))

        #If we've seen data before transform with saved preprocessors
        else:
            
            #Load Column Transformer and Transform data
            ct = load(open('x_transformer.pkl', 'rb'))
            df_processed = ct.transform(features)
               
            #Load Transformer for y
            if y is not None:
                y_scaler = load(open('y_transformer.pkl', 'rb'))
                outputs = y_scaler.transform(outputs)


        # --------------------------------------------------------------------
        # RETURN AS TENSORS    
        x_tensor = torch.tensor(df_processed,dtype = torch.float32)
        
        # check if y is in the data
        if y is not None:
            y_tensor = torch.tensor(y.values,dtype = torch.float32)
        
        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # --------------------------------------------------------------------
        # PRE PROCESS
        #Preprocess input data
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        #Initialize functions to save losses
        #[0,fold,epoch] -> for training losses
        #[1,fold,epoch] -> for validation losses
        rel_losses = np.zeros((2,self.nb_epoch))
        abs_losses= np.zeros((2,self.nb_epoch))
        
        # --------------------------------------------------------------------
        # TRAIN/VALIDATION SPLIT
        #Randomly splits into 90% train and 10% validation 
        train_index,val_index = train_test_split(np.arange(X.shape[0]),
                                                 train_size = 0.9)

        #Data to train
        x_train = X[train_index].detach()
        y_train = Y[train_index].detach()
        
        #Data to evaluate
        x_val = X[val_index].detach()
        y_val = Y[val_index].detach()
        
        # --------------------------------------------------------------------
        # LOAD BATCH TENSOR 
        #To do batching on training data, with shuffling
        torch_dataset_train = data.TensorDataset(x_train,y_train)
        data_loader_train = data.DataLoader(dataset = torch_dataset_train,
                                            batch_size = self.batch_size,
                                            shuffle = True)
        
        # --------------------------------------------------------------------
        # INITIALIZE FUNCTIONS TO SAVE LOSS
        # Moving average over 20 epochs
        N = 20
        cumsum, moving_averages = [0],[]
        old_average = np.inf
        
        # --------------------------------------------------------------------
        # START TRAINING
        for epoch in range(self.nb_epoch):
        
            #Set network to training mode
            self.network.train()
            
            #Batching in every epoch
            for step, (batch_x, batch_y) in enumerate(data_loader_train):
            
                #Clear gradient 
                self.optimizer.zero_grad()
                
                #Forward propagation
                prediction = self.network(batch_x)
                
                #Compute Loss
                loss = nn.MSELoss()(prediction,batch_y)
                
                #Backward propragation    
                loss.backward()
                
                #Update parameters
                self.optimizer.step()
        
            #Evaluate after every epoch
            self.network.eval()
            
            # -----------------------------------------------------------------
            # COMPUTE LOSS
            #Compute total loss on training data
            prediction = self.network.forward(x_train).detach()
            #Absolute training loss
            train_loss_abs = nn.MSELoss()(prediction,y_train)
            abs_losses[0,epoch] = train_loss_abs
            #Relative training loss
            train_loss_rel = torch.sum(torch.div(torch.abs(torch.sub(prediction,y_train)),y_train)) / y_train.shape[0]
            rel_losses[0,epoch] = train_loss_rel

            #Compute total loss on validation data
            prediction = self.network.forward(x_val).detach()
            #Absolute validation loss
            val_loss_abs = nn.MSELoss()(prediction,y_val)
            abs_losses[1,epoch] = val_loss_abs
            #Relative validation loss
            val_loss_rel = torch.sum(torch.div(torch.abs(torch.sub(prediction,y_val)),y_val)) / y_val.shape[0]
            rel_losses[1,epoch] = val_loss_rel   
                             
            #Print losses every 10 folds               
            if ((epoch % 10) == 0):
                if self.verbose:
                    print("EPOCH: {}\t - Training Loss:  {}\t - Validation Loss: {}".format(epoch,train_loss_rel,val_loss_rel))   
            
            #Compute moving average of last N losses
            cumsum.append(cumsum[epoch - 1] + val_loss_abs.numpy())  
            if epoch >= N:
                moving_average = (cumsum[epoch] - cumsum[epoch - N]) / N
                moving_averages.append(moving_average)
            # --------------------------------------------------------------------
            # OPTIONAL: EARLY STOPPING
                #If moving average rising, stop -> not optimal yet
                if (moving_average > old_average):
                    if self.early_stopping:
                        print("Episode: {}\t - Training Loss:  {}\t - Validation Loss: {}".format(epoch,train_loss_rel,val_loss_rel)) 
                        print("Average Validation Loss rising -> break")
                        break
                else:
                    old_average = moving_average
                
        
        self.loss_abs = abs_losses
        self.loss_rel = rel_losses
               
        return self
    

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # --------------------------------------------------------------------
        # PRE PROCESS
        #Preprocess data we want to predict
        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        # --------------------------------------------------------------------
        # EVALUATE
        #Return data we want to evaluate
        self.network.eval()
        prediction = self.network.forward(X).detach().numpy()
        
        
        return prediction

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # --------------------------------------------------------------------
        # PRE PROCESS     
        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        
        #Output from predict is array
        Y = Y.numpy() 
        prediction = self.predict(x)
        
        # --------------------------------------------------------------------
        # EVALUATION SCORES FOR REGRESSION
        # mean squared error
        mse = mean_squared_error(prediction,Y)
        # explained variance
        ex_var = explained_variance_score(prediction,Y)
        # coefficient of determination
        r2 = r2_score(prediction,Y)
        # root mean squared error
        rmse = np.sqrt(mse)
        
        return mse,ex_var,r2, rmse 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x,y,folds = 5): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # --------------------------------------------------------------------    
    # PARAMETER SEARCH SPACE
    params = {
    'lr': [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'epochs': list(range(50,200, 50)),
    'batch_size': [16, 28, 56, 96, 128]}
    epoch = 50
    
    #initialize our return values
    num_folds = 5
    min_error = 1e7
    best_lr = 0
    best_batch = 0
    best_epoch = 0
        
    # --------------------------------------------------------------------
    # KFOLD CROSS VALIDATION
    kfold = KFold(n_splits = num_folds)
    folds = list(kfold.split(np.arange(x.shape[0])))
    
    # --------------------------------------------------------------------
    # BEGIN PARAMETER SEARCH
    for batchsizes in params['batch_size']:
        
        print('---------------------------------------')
        print("Begining training with batch size: " + str(batchsizes))
    
        for learning_rate in params['lr']:
            
            print('---------------------------------------')
            print("Beginning training with learning rate: " + str(learning_rate))
            
            for epoch in params['epochs']:
                print('---------------------------------------')
                print("Begining training with epoch: " + str(epoch))
            
                average_errors = []
                
                for fold, (train_index, val_index) in enumerate(folds):
                
                    # get the data to use for valiation and training
                    x_train = x.iloc[train_index]
                    y_train = y.iloc[train_index]
                    
                    x_val = x.iloc[val_index]
                    y_val = y.iloc[val_index]
        
                    #Call regressor with given parameters
                    regressor = Regressor(x_train, 
                                          nb_epoch = epoch, 
                                          lr = learning_rate, 
                                          batch_size = batchsizes)
                    # Begin training
                    regressor.fit(x_train, y_train)
        
                    # compute the evaluation metrics
                    mse,ex_var,r2,rmse = regressor.score(x_val, y_val)
            
                    # use the mse as threshold
                    fold_error = mse
                    average_errors.append(fold_error)
                average_error = np.average(average_errors)
                # check which one is the best
                if average_error < min_error:
                    # save the best values
                    min_error = average_error 
                    best_lr = learning_rate
                    best_batch = batchsizes
                    best_epoch = epoch
    
    return  best_lr,best_batch, best_epoch

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    
# --------------------------------------------------------------------
# PLOTTING FUNCTIONS
def plot_validation_loss(training_loss, validation_loss):

    title = 'Loss with  Batch Size = 56 and Learning rate = 0.002'
    fig = plt.figure()
    plt.plot(training_loss, label ='Training Loss', color='mediumblue')
    plt.plot(validation_loss, label = 'Validation Loss', color= 'darkorange')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(title+'.png', dpi =250)
    plt.show()
    
def plot_prediction(predicted, y_test):
    fig = plt.figure()
    plt.plot(np.arange(len(predicted)),predicted , label ='Predicted', color='red')
    plt.plot(np.arange(len(y_test)),y_test, label = 'Actual', color= 'black')
    plt.legend()
    plt.xlabel('Sampled Points')
    plt.ylabel('Median House Value')
    plt.title('Predicted Median House Value')
    plt.savefig('predictions.png', dpi =250)
    plt.show()
    

def example_main():

    # --------------------------------------------------------------------
    # LOAD THE DATA
    #define the output label
    output_label = "median_house_value"
    # get the data
    data = pd.read_csv("housing.csv") 
    
    #Randomly shuffle the data
    data = data.sample(frac = 1).reset_index(drop = True)
    
    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]   

    # keep a held out dataset for testing overfitting
    x_train = x[2000:]
    y_train = y[2000:]
    x_test = x_train[0:2000]
    y_test = y_train[:2000]
    
    # --------------------------------------------------------------------
    # TRAIN
    regressor = Regressor(x_train, nb_epoch = 100)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)
    
    # --------------------------------------------------------------------
    # PLOT LOSS
    plot_validation_loss(training_loss= regressor.loss_rel[0,:],
                       validation_loss =regressor.loss_rel[1,:] )
    
    # --------------------------------------------------------------------
    # TEST & PREDICT    
    pred = regressor.predict(x_test)
    
    scaler = load(open('y_transformer.pkl', 'rb'))
 
    # --------------------------------------------------------------------
    # EVALUATE 
    error = regressor.score(x_test, y_test)
    print('--------------------------------------')
    print('Test scores: ')
    print('\nMSE: {} '.format(error[0]))
    print('\nExplained Variance: {}'.format(error[1]))
    print('\nR^2 score: {}' .format(error[2]))
    print('\nRMSE: {}'.format(error[3]))
    print('--------------------------------------')

def main_hyperparameter_search():
    
    # --------------------------------------------------------------------
    # LOAD THE DATA
    # define the output label
    output_label = "median_house_value"
    # get the data
    data = pd.read_csv("housing.csv") 
    #Randomly shuffle the data
    data = data.sample(frac = 1).reset_index(drop = True)
    
    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    # keep a held out dataset for testing overfitting
    x_train = x[2000:].reset_index(drop = True)
    y_train = y[2000:].reset_index(drop = True)
    x_test = x_train[0:2000].reset_index(drop = True)
    y_test = y_train[:2000].reset_index(drop = True)
    
    # --------------------------------------------------------------------
    # BEGIN HYPERPARAMETER SEARCH
    parameters = RegressorHyperParameterSearch(x_train,y_train)
    # return the best parameters
    best_lr,best_batch, best_epoch = parameters
    
    print('--------------------------------------')
    print('BEST RESULTS FROM HYPERPARAMETER SEARCH')
    print('Best learning Rate: {}'.format(best_lr))
    print('Best Batch Size: {}'.format(best_batch))
    print('Best Epochs: {}'.format(best_epoch))
    
    # use the parameters to train
    regressor = Regressor(x_train, nb_epoch = best_epoch, 
                          lr = best_lr, batch_size=best_batch)
    # --------------------------------------------------------------------
    # TRAIN
    regressor.fit(x_train, y_train)
    
    # --------------------------------------------------------------------
    # tEST & PREDICT
    pred = regressor.predict(x_test)

    # Error
    error = regressor.score(x_test, y_test)
    print('--------------------------------------')
    print('Test scores: ')
    print('\nMSE: {} '.format(error[0]))
    print('\Explained Variance: {}'.format(error[1]))
    print('\R^2 score: {}' .format(error[2]))
    print('\RMSE: {}'.format(error[3]))
    print('\n--------------------------------------')
        
    
class Network(nn.Module):
    
    def __init__(self,input_dim,output_dim, neurons = [256,256,256,256,1],activations = ['relu','relu','relu','relu,','identity']):
        
        super(Network,self).__init__()
        
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons = neurons
        self.activations = activations
        
        
        self._layers = nn.ModuleList()
    
        n_in = self.input_dim

        for i in range(len(self.neurons)):
            
            self._layers.append(nn.Linear(in_features = n_in,out_features = self.neurons[i]))
            n_in = self.neurons[i]
        
        # add a dropout layer
        #self.dropout = nn.Dropout(p=1)

        
    def forward(self,input):
        
        outputs = input
        
        for i in range(len(self.activations)):
            if (self.activations[i] == 'relu'):
                outputs = torch.nn.functional.relu(self._layers[i](outputs))
            elif (self.activations[i] == 'sigmoid'):
                outputs = torch.nn.functional.sigmoid(self._layers[i](outputs))
            elif (self.activations[i] == 'identity'):
                outputs = self._layers[i](outputs)
        
        #outputs = self.dropout(outputs)
        return outputs

if __name__ == "__main__":
    example_main()
    #main_hyperparameter_search()
    
    



        
       

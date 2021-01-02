def RegressorHyperParameterSearch(x, y, num_folds = 5): 
    #the input here should be the training data from our original split
    # we now use this training data for 5 fold cross validation on each combination of hyperparameters
    
    
    
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
    
    #the parameters we are testing
    params = {
    'lr': [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_epochs': list(range(500,5500,500))}
        
    all_errors = []
    
    #initialize our return values
    min_error = 1e7
    best_lr = 0
    best_epoch = 0
    
    out_csv = 'hyperparametertuning.csv'
        
    #rows for our data frame
    rows = []
    
        

    for learning_rate in params[lr]:
        # we first iterate over learning rates 
        print("training on learning rate " + str(learning_rate))

        for epoch in params[max_epochs]:
            print("training on epoch " + str(epoch))

            average_errors = []

            # we now do 5 fold X validation for this learning rate + epoch combo 
                       
            kfold = KFold(n_splits = num_folds)
            folds = kfold.split(np.arange(x.shape[0]))
            
            for fold, (train_index, val_index) in enumerate(folds):

                #Data to train
                x_train = x[train_index].detach()
                y_train = y[train_index].detach()

                #Data to evaluate
                x_val = x[val_index].detach()
                y_val = y[val_index].detach()
       
                row = []
               
            
            
            
            
            
                #TODO 
                #HERE this assumes our fit function does not perform X validation inside of it

                regressor = Regressor(x_train, nb_epoch = epoch, lr = learning_rate)
                regressor.fit(x_train, y_train)
                
                #evaluate the error with our validation set 
                error = regressor.score(x_val, y_val)
                average_errors.append(error)
            
            
            
            
            
            
            # our average error for 5 fold cross validation on this combo of HPs
            average_error = np.average(average_errors)
            #check if less than our minimum running average 
            if average_error < min_error:
                min_error = error 
                best_lr = learning_rate
                best_epoch = epoch
                
            row.append(average_error)
        
            print("For learning rate " + str(learning_rate) + "and epoch" + str(epoch) + "the error is" +str(error))
            print("So far, our best error is" + str(best_error) + "for learning rate" + str(learning_rate) + "and epoch" + str(epoch))
            print("\n")
        
        rows.append(row)


    df = pd.DataFrame(rows, columns=params[max_epochs], index = params[lr])

    
    df.to_csv(out_csv)
                                                                       
    return best_lr, best_epoch
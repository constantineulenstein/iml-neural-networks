# IML_Artificial_Neural_Networks

## Description:

This project contains the Introduction to Machine Learning Coursework 2 Neural Networks implementation by Daphne Demekas, Nasma Dasser, Constantin Eulenstein and Kevin Landert. The project is split into two parts. In the first part, we created a neural network mini-library using only ```NumPy```. In the second part we designed an algorithm using Neural Networks and ```PyTorch``` to predict the median house price in California. The report for the project is available under the _Intro_ML_Coursework2_report.pdf_ file.


## Project structure:
* **PART1**: 
    * _part1_nn_library.py_: contains the implementation of a modular multi-layer neural network library 
    * _IRIS.dat_: file used for testing the neural network library
* **PART2**:
    * _part2_house_value_regression.py_: contains the implementation of a neural network architecture to predict the median house price in California
    * _housing.csv_: California Housing Dataset 

* Additionally, we have included in **jupyter_lab** notebooks to run the models and in **data_analysis** functions used in the report for plotting and in **legacy** legacy functions. 

## Requirements
IML_Artificial_Neural_Networks requires the following to run: 
* ```Python3```
* ```numpy ```
* ```matplotlib```
* ```PyTorch```
* ```scikit-learn```

## Run instructions:

* **PART1** can be run via the command line ```python3 part1_nn_library.py``` . 
    * The Code is then tested on the _IRIS.dat_ dataset using Relu activation function
    * This returns the predictions, as well as the training and validation loss and the accuracy based on a test data split from the IRIS dataset.
    * Please make sure that the _IRIS.dat_ for testing is in the same folder as the file 
* **PART2** can be run via the command line ```python3 part2_house_value_regression.py``` .
    * This runs the main algorithm and returns the predicted median house value 
    * By uncommenting line 700 of _python3 part2_house_value_regression.py_ you can run the hyperparameter search 
    * Please make sure that the _housing.csv_ for training is in the same folder as the file 
    * To **test** part2 with another file:
        1. Upload new housing dataset in the same folder
        2. In the _example_main() function upload the test set
## Credit
Daphne Demekas, Nasma Dasser, Constantin Eulenstein and Kevin Landert

## URL for the project on GitLab

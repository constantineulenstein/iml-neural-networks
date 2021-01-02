#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:57:29 2020

@author: nasmadasser
"""
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv("housing.csv")

printit=False
# ----------------------------------------------------------------------------
# FULL HISTOGRAM
if printit:
    title ='Summary statistics for the California Hosuing dataset'
    data.hist(figsize=(20,15), color = 'grey', grid=False,rwidth=0.9 )
    plt.savefig('./report/'+title +'.png', dpi=250)
    plt.show()

# ----------------------------------------------------------------------------
## DISTRIBUTION CATEGORICAL FEATURES
def plot_cat():
    plt.rcParams['font.size'] = '12'
    plt.figure(figsize=(7,8))
    title='Distribution of the ocean_proximity feature'
    sns.countplot(data.ocean_proximity, palette = 'hls')
    #plt.title(title )
    plt.savefig ('./report/'+title +'.png', dpi=250)
    plt.show()
plot_cat()

# ----------------------------------------------------------------------------
## MISSING VALUES
if printit:
    print(data.isnull().sum())

# ----------------------------------------------------------------------------
## SUMMARY STATISTICS
if printit:
    stats = data.describe()
    print(data.describe())

# ----------------------------------------------------------------------------
# CATEGORICAL TO DUMMY VARIABLE
#this gets rid of the prefix
new_data = pd.get_dummies(data.ocean_proximity)

data[new_data.columns] = new_data

data = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
        , '<1H OCEAN', 'INLAND',
        'ISLAND', 'NEAR BAY', 'NEAR OCEAN','median_house_value']]

# ----------------------------------------------------------------------------
# CORRELATION PLOT
def plot_corr():
    mask = np.zeros_like(data.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plt.rcParams['font.size'] = '12'
    
    title = 'Correlation between features in the dataset'
    plt.figure(figsize=(8,7))
    sns.heatmap(data.corr(), mask=mask, vmax=.3, center=0, cmap="gist_earth",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    #plt.title(title)
    plt.savefig ('./report/'+title +'.png',bbox_inches='tight',dpi=250)
    
plot_corr()

# ----------------------------------------------------------------------------
# PLOT PREDICTION
def plot_prediction():
    from matplotlib.ticker import FuncFormatter
    
    
    def tausands(x, pos):
        return '%1.0fK' % (x * 1e-3)
    y_test = pd.read_csv('y_test.csv', index_col=0)
    prediction = pd.read_csv('prediction.csv', index_col=0)
    
    formatter = FuncFormatter(tausands)
    
    fig, ax = plt.subplots()
    plt.rcParams['patch.facecolor'] = 'b'
    title ='Actual vs predicted results'
    sns.distplot(y_test, color = 'grey', label= 'Actual')
    sns.distplot(prediction, color='green', label ='Predicted')
    plt.title(title)
    ax.xaxis.set_major_formatter(formatter)
    plt.xlabel('median house value ($)')
    plt.ylabel('density')
    plt.savefig('./report/'+title +'.png',bbox_inches='tight',dpi=250)
    plt.legend(frameon=False)

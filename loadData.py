# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:06:16 2020

@author: Alex
"""

# import necessary libraries
from sklearn.datasets import load_svmlight_file as loadSVM
import scipy.sparse
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from word2number import w2n
import os
import time as time
from datetime import datetime, timedelta 
from sklearn.datasets import dump_svmlight_file

os.chdir(r'C:\Users\Alex\Documents\GitHub\oldbailey')

# import custom learning libraries
from perceptrons import *
from svm import *

def roundup(x, by):
    return int(math.ceil(x / float(by))) * by

#%% import data

def loadBOW():
    '''
    load bag-of-words dataset.

    Returns
    -------
    X_bowTrn : df
        training bag-of-words features.
    y_bowTrn : df
        training bag-of-words labels.
    X_bowTst : df
        testing bag-of-words features.
    y_bowTst : df
        testing bag-of-words labels.
    majLbl : dict
        majority label for test and training datasets. learner should improve upon majority label.
    '''   

    X_bowTrn, y_bowTrn = loadSVM('project_data/data/bag-of-words/bow.train.libsvm')
    X_bowTrn = pd.DataFrame.sparse.from_spmatrix(X_bowTrn)
    y_bowTrn = pd.DataFrame(y_bowTrn, columns=['Label'])
    
    X_bowTst, y_bowTst = loadSVM('project_data/data/bag-of-words/bow.test.libsvm')
    X_bowTst = pd.DataFrame.sparse.from_spmatrix(X_bowTst)
    y_bowTst = pd.DataFrame(y_bowTst, columns=['Label'])
    
    X_bowEval, y_bowEval = loadSVM('project_data/data/bag-of-words/bow.eval.anon.libsvm')
    X_bowEval = pd.DataFrame.sparse.from_spmatrix(X_bowEval)
    
    majLbl = {}
    majLbl['TstLblP']  = np.mean(y_bowTst.values)
    majLbl['TstLbl'] = int(np.round(majLbl['TstLblP']))
    majLbl['TrnLblP'] = np.mean(y_bowTrn.values)
    majLbl['TrnLbl'] = int(np.round(majLbl['TrnLblP']))
    
    return X_bowTrn, y_bowTrn, X_bowTst, y_bowTst, X_bowEval, y_bowEval, majLbl

X_bowTrn, y_bowTrn, X_bowTst, y_bowTst, X_bowEval, y_bowEval, majLbl = loadBOW()

#%% load glove

def loadGlo():
    '''
    load glove dataset.
    '''   

    X_gloTrn, y_gloTrn = loadSVM('project_data/data/glove/glove.train.libsvm')
    X_gloTrn = pd.DataFrame.sparse.from_spmatrix(X_gloTrn)
    y_gloTrn = pd.DataFrame(y_gloTrn, columns=['Label'])
    
    X_gloTst, y_gloTst = loadSVM('project_data/data/glove/glove.test.libsvm')
    X_gloTst = pd.DataFrame.sparse.from_spmatrix(X_gloTst)
    y_gloTst = pd.DataFrame(y_gloTst, columns=['Label'])
    
    X_gloEval, y_gloEval = loadSVM('project_data/data/glove/glove.eval.anon.libsvm')
    X_gloEval = pd.DataFrame.sparse.from_spmatrix(X_gloEval)
    
    majLbl = {}
    majLbl['TstLblP']  = np.mean(y_gloTst.values)
    majLbl['TstLbl'] = int(np.round(majLbl['TstLblP']))
    majLbl['TrnLblP'] = np.mean(y_gloTrn.values)
    majLbl['TrnLbl'] = int(np.round(majLbl['TrnLblP']))
    
    return X_gloTrn, y_gloTrn, X_gloTst, y_gloTst, X_gloEval, y_gloEval, majLbl

X_gloTrn, y_gloTrn, X_gloTst, y_gloTst, X_gloEval, y_gloEval, _ = loadGlo()

#%% load meta data and select features

def loadMeta(features, oneHot):
    '''
    load meta data with select features and one-hot encoding

    Parameters
    ----------
    features : list
        DESCRIPTION.
    oneHot : str
        select 'custom' or 'full' one-hot encoding of metadata.

    Returns
    -------
    metaTrn : df
        training metadata.
    metaTst : df
        testing metadata.
    '''        
    
    metaTrn = pd.read_csv('project_data/data/misc-attributes/misc-attributes-train.csv')
    metaTst = pd.read_csv('project_data/data/misc-attributes/misc-attributes-test.csv')

    #% select meta data features 
    metaTrn = metaTrn[features]
    metaTst = metaTst[features]

    # convert defendant age strings to integer & bin defendant ages
    errCnt = 0;
    for idx, row in metaTrn.iterrows():
    
        # convert defendant age strings to integer    
        try:
            metaTrn.at[idx,'defendant_age'] = w2n.word_to_num(str(row.defendant_age))                
        except ValueError:        
            errCnt+=1 
            metaTrn.at[idx,'defendant_age'] = 0
        
    # roundup to nearest 10
    metaTrn.at[idx,'defendant_age'] = roundup(metaTrn.at[idx,'defendant_age'],10)
    print('Number of errors:', errCnt)     

    if oneHot == 'custom':
        for f in features:
            if f == 'defendant_gender': #if def_gender = male -> assign 1  
              # train data  
                metaTrn[f] = metaTrn[f].replace({'male': 1, 'female': 0, 'indeterminate': 0})
              # test data  
                metaTst[f] = metaTst[f].replace({'male': 1, 'female': 0, 'indeterminate': 0})
            elif f == 'num_victims': #if crime = theft -> assign 1          
              # train data     
                metaTrn[f].loc[metaTrn[f] != 1] = 0 # crimes with != 1 get label value 0 (> 13,000 single victime crimes.)
              # test data  
                metaTst[f].loc[metaTst[f] != 1] = 0 
            elif f == 'offence_category': #if crime = theft -> assign 1
              # train data
                metaTrn[f] = metaTrn[f].replace({'theft': 1})
                metaTrn[f].loc[metaTrn[f] != 1] = 0 
                metaTrn[f] = metaTrn[f].astype(object).astype(int)
              # test data
                metaTst[f] = metaTst[f].replace({'theft': 1})
                metaTst[f].loc[metaTst[f] != 1] = 0 
                metaTst[f] = metaTst[f].astype(object).astype(int)
            elif f == 'victim_genders': #if victim = male -> assign 1
              # train data
                metaTrn[f] = metaTrn[f].replace({'male': 1})
                metaTrn[f].loc[metaTrn[f] != 1] = 0 
                metaTrn[f] = metaTrn[f].astype(object).astype(int)
              # test data
                metaTst[f] = metaTst[f].replace({'male': 1})
                metaTst[f].loc[metaTst[f] != 1] = 0 
                metaTst[f] = metaTst[f].astype(object).astype(int)

    elif oneHot == 'full':# full onehot encoding            
        metaTrn = pd.get_dummies(metaTrn,drop_first=True)
        metaTst = pd.get_dummies(metaTst,drop_first=True)

        return metaTrn, metaTst

# select features for meta data
sel_features = ['defendant_age','num_victims','defendant_gender','offence_category'];
metaTrn, metaTst = loadMeta(sel_features, 'full')
         
#%% build augmented training and testing input datasets

def augData(metaTrn, X_trn, y_trn, metaTst, X_tst, y_tst):
    '''   
    combine core dataset features with meta data.

    Parameters
    ----------
    metaTrn : df
       training metadata.
    X_trn : df
        training dataset features.
    y_trn : df
        training dataset labels.
    metaTst : df
        testing metadata.
    X_tst : df
        testing dataset features.
    y_tst : df
        testing dataset labels.

    Returns
    -------
    train_in : df
        final training dataset augumented with meta data.
    test_in : df
        final training dataset augumented with meta data.
    '''

    # count the number of non-zero values as an indicator of a feature's importance
    #sig = pd.DataFrame(np.count_nonzero(X_trn, axis=0)/X_trn.shape[1])
    #sig = sig[sig.values>0.01] # index of significant features
    #X_trn = X_trn[sig.index];
    
    X_trn = pd.concat([X_trn, metaTrn], axis=1, sort=False)        
    train_in = pd.concat([y_trn, X_trn], axis=1, sort=False) # add label to features
       
    X_tst = pd.concat([X_tst, metaTst], axis=1, sort=False)
    test_in = pd.concat([y_tst, X_tst], axis=1, sort=False)
    
    train_in.Label = train_in['Label'].replace(0,-1) #relabel to align with perceptron
    test_in.Label = test_in['Label'].replace(0,-1)
    
    print('Train Input:',train_in.shape)
    print('Test Input:',test_in.shape)  
    
    return train_in, test_in, sig

#train_in, test_in = augData(metaTrn, X_bowTrn, y_bowTrn, metaTst, X_bowTst, y_bowTst)

X_trnComb = pd.concat([X_bowTrn, X_gloTrn], axis=1, sort=False)
X_trnComb.columns= np.arange(0,X_trnComb.shape[1])

train_in, test_in, sig = augData(metaTrn, X_trnComb, y_bowTrn, metaTst, X_bowTst, y_bowTst)

#%%

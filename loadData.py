# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:06:16 2020

@author: Alex
"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from word2number import w2n
import os
from sklearn.datasets import dump_svmlight_file

def roundup(x, by):
    return int(math.ceil(x / float(by))) * by

os.chdir(r'C:\Users\Alex\Documents\GitHub\oldbailey')

#%% import data

from sklearn.datasets import load_svmlight_file as loadSVM
import scipy.sparse

def loadBOW():

    X_bowTrn, y_bowTrn = loadSVM('project_data/data/bag-of-words/bow.train.libsvm')
    X_bowTrn = pd.DataFrame.sparse.from_spmatrix(X_bowTrn)
    y_bowTrn = pd.DataFrame(y_bowTrn, columns=['Label'])
    
    X_bowTst, y_bowTst = loadSVM('project_data/data/bag-of-words/bow.test.libsvm')
    X_bowTst = pd.DataFrame.sparse.from_spmatrix(X_bowTst)
    y_bowTst = pd.DataFrame(y_bowTst, columns=['Label'])
    
    majLbl = {}
    majLbl['TstLblP']  = np.mean(y_bowTst.values)
    majLbl['TstLbl'] = int(np.round(majLbl['TstLblP']))
    majLbl['TrnLblP'] = np.mean(y_bowTrn.values)
    majLbl['TrnLbl'] = int(np.round(majLbl['TrnLblP']))
    
    return X_bowTrn, y_bowTrn, X_bowTst, y_bowTst, majLbl

X_bowTrn, y_bowTrn, X_bowTst, y_bowTst, majLbl = loadBOW()

#%% load meta data and select features

def loadMeta(features, oneHot):
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
    X_Trn = pd.concat([X_trn, metaTrn], axis=1, sort=False)
    train_in = pd.concat([y_trn, X_Trn], axis=1, sort=False)
    
    X_Tst = pd.concat([X_tst, metaTst], axis=1, sort=False)
    test_in = pd.concat([y_tst, X_Tst], axis=1, sort=False)
    
    train_in.Label = train_in['Label'].replace(0,-1) #relabel to align with perceptron
    test_in.Label = test_in['Label'].replace(0,-1)
    
    print('Train Input:',train_in.shape)
    print('Test Input:',test_in.shape)  
    
    return train_in, test_in

train_in, test_in = augData(metaTrn, X_bowTrn, y_bowTrn, metaTst, X_bowTst, y_bowTst)



#%% 

from perceptrons import *

# prepare best hyper-parameters
bestHP = pd.DataFrame(np.zeros((4,6)), index=['std','decay','avg','margin'],
                                    columns=['fold','rate','margin','updates','trn_accuracy', 'val_accuracy'])
weights = {}; biases = {}; 
acc_s0 = acc_d0 = acc_a0 = acc_m0 = 0; # initialize baseline accuracy value
up_s0 = up_d0 = up_a0 = up_m0 = 0; # initialize update counts 
lc = {}
    
for f in np.arange(1, n_folds+1):
    up_s0 = up_d0 = up_a0 = up_m0 = 0; # initialize updates counter for each fold
    print('Fold - ', f)
    
    data_fold = dataCV[f]
    
    # initialize parameters
    etas = [1,0.5,0.1] # learning rates 
    
    # initialize weights and bias terms
    predAcc_dict = {}; # store accuracies
    w0 = np.random.uniform(-0.01, 0.01, size=(data_fold['trn'].shape[1]-1)) 
    b0 = np.random.uniform(-0.01, 0.01)
    T = 10; # number of epochs

    # --- average
    print('<<< Averaging Perceptron >>>')
    for r in etas:
        print('  Learning rate:', r)
        w_avg, b_avg, _, lc[r], _, _ = perc_avg(data_fold['trn'],w0,b0,r,T)
        #trnAcc_a = pred_acc('Avg - Training', data_fold['trn'], w_avg, b_avg) 
        valAcc_a = pred_acc('Avg - Validation', data_fold['val'], w_avg, b_avg) 
    
        if valAcc_a > acc_a0:
            print('\n-Batch Perceptron - Averaging:', r)
            print('-Update', np.round(acc_a0,3), '->', np.round(valAcc_a,3), f, r, 0)
            acc_a0 = valAcc_a; # if predicted accuracy for these hp is better, update
            up_a0 += 1;
            bestHP.loc['avg'] = [f, r, 0, up_a0, 'na', valAcc_a]
            weights['avg'] = w_avg;
            biases['avg'] = b_avg;   

#%%

weights['avg'].to_csv('weights.csv')


#%% make predictions

def makePred(data, w, b):

    wT = w.transpose();
    yi_p = [];
    
    for ix, row in data.iterrows():
        xi = row;
        
        if np.dot(wT,xi) + b >= 0: # create predicted label
            yi_p.append(1) # true label
        else: yi_p.append(0) # false label #NOTE check label true/false [1,0] or [1,-1]

    return yi_p

yi_predict = makePred(eval_in, weights['avg'], biases['avg'])

print(np.mean(yi_predict))

#%%

w_eval = weights['avg'][:-19] #remove meta data training weights
wT = w_eval.transpose(); 

b = biases['avg'];
yi_p = [];

for ix, row in eval_in.iterrows():
    xi = row;
    val = np.dot(wT,xi) + b
    
    if np.dot(wT,xi) + b >= 0: # create predicted label
        yi_p.append(1) # true label
    else: yi_p.append(0) # false label #NOTE check label true/false [1,0] or [1,-1]

print(np.mean(yi_p))

yi_p = pd.DataFrame(yi_p, columns=['Predicted Label'])
yi_p.to_csv('yi_p.csv')

#%% prediction on eval data

X_bowEval = loadSVM('project_data/data/bag-of-words/bow.eval.anon.libsvm')
eval_in = pd.DataFrame.sparse.from_spmatrix(X_bowEval[0])

#%% tryout learning

#from perceptrons import perc_avg, pred_acc
from perceptrons import *

data_fold = dataCV[4]
r = 0.5; T = 10;
w0 = np.random.uniform(-0.01, 0.01, size=(data_fold['trn'].shape[1]-1)) 
b0 = np.random.uniform(-0.01, 0.01)
print('<<< Averaging Perceptron >>>')
#w_avg, b_avg, _, _, _, _ = perc_avg(data_fold['trn'],w0,b0,r,T)

#def perc_std(data,w,b,r,T):      

w = w0; # initialize values
wT = w.transpose(); 
b = b0;

data = data_fold['trn']
data.Label = data['Label'].replace(0,-1)

wT = w.transpose(); w_sum = w; b_sum = b; s = 0; # initialize values
acc0 = 0 # initialize accuracy baseline
lc = np.zeros((T)) # learning curve

for ep in range(T):   
    up = 0; # initialize update count
    print('Epoch', ep)
    
    for ix, row in data.iterrows():
        yi = row.Label # select sample label
        xi = row.drop('Label') # select sample features
        
        if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU        
            w += r * yi * xi # update weight matrix
            b += r * yi # update bias term
            up += 1;
            wT = w.transpose()
            if np.mod(up,1000) == 0:
                print(' ', up)
        
        # accumulate weights
        w_sum += w; b_sum += b;
        s += 1;
    
    w_avg = w_sum/s; # average weight
    b_avg = b_sum/s; # average bias    
    
    # store best accuracy from epochs
    epAcc = pred_acc('Average',data,w_avg,b_avg) 
    lc[ep] = epAcc;
    if epAcc > acc0: 
        best_epAcc = [ep, epAcc, up];
        w_best = w_avg; b_best = b_avg;
        acc0 = epAcc;

plt.plot(lc)
#%%

wT = w_avg.transpose(); b = b_avg;
yi_p = []; acc_cnt = 0;

for ix, row in data.iterrows():
    yi = row.Label # select sample label
    xi = row.drop('Label') # select sample features
    val = np.dot(wT,xi) + b
    print(val)
    
    if val >= 0: # create predicted label
        yi_p.append(1) # true label
    else: yi_p.append(0) # false label #NOTE check label true/false [1,0] or [1,-1]

    if yi_p[-1] == yi:
        acc_cnt += 1; # count correct labels
       
acc = acc_cnt/len(data)            
print('- Pred accuracy: {:.4f}'.format(acc))  


#%%

trnAcc_a = pred_acc('Averaging - Training', data_fold['trn'], w_avg, b_avg) 
testAcc_a = pred_acc('Averaging - Testing', data_fold['val'], w_avg, b_avg) 
        
#%%


wT = w.transpose();
yi_p = []; acc_cnt = 0;

for ix, row in data.iterrows():
    yi = row.Label # select sample label
    xi = row.drop('Label') # select sample features
    
    if np.dot(wT,xi) + b >= 0: # create predicted label
        yi_p.append(1) # true label
    else: yi_p.append(-1) # false label

    if yi_p[-1] == yi:
        acc_cnt += 1; # count correct labels
       
acc = acc_cnt/len(data)            
print('- Pred accuracy: {:.4f}'.format(acc))  



#%% rename columns
cols = list(dataCV[1]['val'])
cols[0] = 'Label'
for f in np.arange(1, n_folds+1):
    
    dataCV[f]['val'].columns = cols
    dataCV[f]['trn'].columns = cols

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
import os

def roundup(x, by):
    return int(math.ceil(x / float(by))) * by

os.chdir(r'C:\Users\Alex\Documents\GitHub\oldbailey')

#%% import data

from sklearn.datasets import load_svmlight_file as loadSVM
import scipy.sparse

X_bowTrn, y_bowTrn = loadSVM('project_data/data/bag-of-words/bow.train.libsvm')
X_bowTrn = pd.DataFrame.sparse.from_spmatrix(X_bowTrn)
y_bowTrn = pd.DataFrame(y_bowTrn, columns=['Label'])

X_bowTst, y_bowTst = loadSVM('project_data/data/bag-of-words/bow.test.libsvm')
X_bowTst = pd.DataFrame.sparse.from_spmatrix(X_bowTst)
y_bowTst = pd.DataFrame(y_bowTst, columns=['Label'])

majLbl = {}
majLbl['TstLblP'] = np.mean(y_bowTst.values)
majLbl['TstLbl'] = int(np.round(majLbl['TstLblP']))
majLbl['TrnLblP'] = np.mean(y_bowTrn.values)
majLbl['TrnLbl'] = int(np.round(majLbl['TrnLblP']))

#%% load meta data and select features

metaTrn = pd.read_csv('project_data/data/misc-attributes/misc-attributes-train.csv')
metaTst = pd.read_csv('project_data/data/misc-attributes/misc-attributes-test.csv')

#%% descriptive statistics - defendant age

maxAge = roundup(np.max(metaTrn['defendant_age']),10)

sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(8,5), dpi=240)
sns.distplot(metaTrn['defendant_age'].where(metaTrn['defendant_age'] > 0).dropna(),
    bins = np.arange(0,maxAge,5), label='Known Ages')
sns.distplot(metaTrn['defendant_age'].where(metaTrn['defendant_age'] == 0).dropna(),
    bins = np.arange(0,maxAge,5), label='Unknown Ages')
#metaTrn['defendant_age'].where(metaTrn['defendant_age'] > 0).dropna().plot.hist(
#    bins = np.arange(0,maxAge,10), alpha=0.8, rwidth=0.95, label='Known Ages')
#metaTrn['defendant_age'].where(metaTrn['defendant_age'] == 0).dropna().plot.hist(
#    bins=np.arange(0,20,10), alpha=0.8, rwidth=0.95, label='Unknown Ages')

plt.title('Defendant Age Distribution')
plt.xlim([-0.5,maxAge])
plt.legend()

#%% descriptive statistics - defendant age

maxVictim = roundup(np.max(metaTrn['num_victims']),5)

sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(8,5), dpi=240)
metaTrn['num_victims'].plot.hist(density=True, bins=np.arange(0,maxVictim,1))

plt.title('Number of Victims Distribution')
plt.legend()

#%% select meta data features 
select_features = ['defendant_age','defendant_gender','offence_category']
metaTrn = metaTrn[select_features]
metaTst = metaTst[select_features]

#%% bin defendant ages

from word2number import w2n

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
        

#%% categorize 

featureize = {}
for f in list(metaTrn):
    featureize[f] = metaTrn[f].value_counts()
    
    #featureize[f+'_count'] = len(set(metaTrn[f]))
    #featureize[f+'_set'] = list(set(metaTrn[f]))

#featureize = pd.DataFrame()

#%%

metaTrn = pd.read_csv('project_data/data/misc-attributes/misc-attributes-train.csv')
metaTst = pd.read_csv('project_data/data/misc-attributes/misc-attributes-test.csv')

# feature types
featureize = {}
for f in list(metaTrn):
    featureize[f] = metaTrn[f].value_counts()

# custom one-hot
select_features = ['defendant_gender','num_victims','offence_category','victim_genders']
metaTrn = metaTrn[select_features]
metaTst = metaTst[select_features]

for f in select_features:
    if f == 'defendant_gender':      
      # train data  
        metaTrn[f] = metaTrn[f].replace({'male': 1, 'female': 0, 'indeterminate': 0})
      # test data  
        metaTst[f] = metaTst[f].replace({'male': 1, 'female': 0, 'indeterminate': 0})
    elif f == 'num_victims':           
      # train data     
        metaTrn[f].loc[metaTrn[f] != 1] = 0 # crimes with != 1 get label value 0 (> 13,000 single victime crimes.)
      # test data  
        metaTst[f].loc[metaTst[f] != 1] = 0 
    elif f == 'offence_category':
      # train data
        metaTrn[f] = metaTrn[f].replace({'theft': 1})
        metaTrn[f].loc[metaTrn[f] != 1] = 0 
        metaTrn[f] = metaTrn[f].astype(object).astype(int)
      # test data
        metaTst[f] = metaTst[f].replace({'theft': 1})
        metaTst[f].loc[metaTst[f] != 1] = 0 
        metaTst[f] = metaTst[f].astype(object).astype(int)
    elif f == 'victim_genders':
      # train data
        metaTrn[f] = metaTrn[f].replace({'male': 1})
        metaTrn[f].loc[metaTrn[f] != 1] = 0 
        metaTrn[f] = metaTrn[f].astype(object).astype(int)
      # test data
        metaTst[f] = metaTst[f].replace({'male': 1})
        metaTst[f].loc[metaTst[f] != 1] = 0 
        metaTst[f] = metaTst[f].astype(object).astype(int)
        
#%% build augmented training and testing input datasets
X_Trn = pd.concat([X_bowTrn, metaTrn], axis=1, sort=False)
train_in = pd.concat([y_bowTrn, X_Trn], axis=1, sort=False)

X_Tst = pd.concat([X_bowTst, metaTst], axis=1, sort=False)
test_in = pd.concat([y_bowTst, X_Tst], axis=1, sort=False)

train_in.Label = train_in['Label'].replace(0,-1) #relabel to align with perceptron
test_in.Label = test_in['Label'].replace(0,-1)

print('Train Input:',train_in.shape)
print('Test Input:',test_in.shape)  

#%% onehot encoding    
metaTrn = pd.get_dummies(metaTrn,drop_first=True)

# describe meta training data
metaTrn.describe()
metaTrn_totals = np.sum(metaTrn)

#%% Crime Distribution

crimes = ['Damage', 'Deception', 'Kill', 'Misc', 'Royal Offense', 'Sexual', 'Theft', 'Violent Theft']
metaTrn_crimes = metaTrn_totals[-8:]
metaTrn_crimes.index = crimes

sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(8,5), dpi=240)
metaTrn_crimes.plot.bar()
plt.title('Crime Comparison')
plt.ylabel('Count')

#%%
cols = list(train_in)
cols[0] = 'Label'
train_in.columns = cols
cols = list(test_in)
cols[0] = 'Label'
test_in.columns = cols

train_in.Label = train_in['Label'].replace(0,-1)
test_in.Label = test_in['Label'].replace(0,-1)

#%% filter 

dump_svmlight_file(d,e,'C:/result/smvlight2.dat')

#%% cross-validation
'''
For this problem we will implement k-folds cross-validation.
'''

from sklearn.datasets import dump_svmlight_file

n_folds = 5
k = int(len(train_in)/n_folds); #samples per validation fold

ids = list(np.arange(len(train_in))) # create list of indeces
np.random.shuffle(ids) # randomly shuffle for CV

train_arr = np.array(train_in)
test_arr = np.array(test_in)

#dataCV = {};
for f in range(0,n_folds):
    
    print('+++ CV Data Fold -', f,' +++')      
    #dataCV[f] = {};
    
    # create validation fold
    idxVal = ids[f*k:f*k+k]
    idxTrn = list(set(ids)-set(idxVal))
    np.random.shuffle(idxTrn)
    
    #dataCV[f]['val'] = train_arr[idxVal]
    #dataCV[f]['trn'] = train_arr[idxTrn]
    arrFold_val = train_arr[idxVal]
    arrFold_trn = train_arr[idxTrn]
    
    
    filename = 'working_data/fold'+str(f)+'_val.dat'
    dump_svmlight_file(arrFold_val[:,1:], arrFold_val[:,0], filename)
    filename = 'working_data/fold'+str(f)+'_trn.dat'
    dump_svmlight_file(arrFold_trn[:,1:], arrFold_trn[:,0], filename)
    #dataCV[f]['val'] = np.array(train_in.iloc[idxVal])
    #dataCV[f]['trn'] = np.array(train_in.iloc[idxTrn])
    
    # save csvs
    #fileVal = 'working_data/fold'+str(fold)+'_val.csv'
    #dataCV[fold]['val'].to_csv(fileVal)
    #fileTrn = 'working_data/fold'+str(fold)+'_trn.csv'
    #dataCV[fold]['trn'].to_csv(fileTrn)

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

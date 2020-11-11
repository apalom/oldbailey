# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:13:26 2020

@author: Alex
"""


#%% cross-validation
'''
For this problem we will implement k-folds cross-validation.
'''

from sklearn.datasets import dump_svmlight_file

#% create folder
date_folder = os.path.join(os.getcwd(), 'working_data',
                           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

#% cross-validation
n_folds = 2
k = int(len(train_in)/n_folds); #samples per validation fold

ids = list(np.arange(len(train_in))) # create list of indeces
np.random.shuffle(ids) # randomly shuffle for CV

train_arr = np.array(train_in)
test_arr = np.array(test_in)

dataCV = {};
for f in range(0,n_folds):
    
    print('+++ CV Data Fold -', f,' +++')      
    dataCV[f] = {}; # create fold
    
    # create validation fold
    idxVal = ids[f*k:f*k+k]
    idxTrn = list(set(ids)-set(idxVal))
    np.random.shuffle(idxTrn)
    
    # assign validation and training data for each fold
    dataCV[f]['val'] = train_arr[idxVal]
    dataCV[f]['trn'] = train_arr[idxTrn]
    
    arrFold_val = train_arr[idxVal]
    arrFold_trn = train_arr[idxTrn]    
    
    filename = os.path.join(date_folder,'fold'+str(f)+'_val.dat')
    dump_svmlight_file(arrFold_val[:,1:], arrFold_val[:,0], filename)
    filename = os.path.join(date_folder,'fold'+str(f)+'_trn.dat')
    dump_svmlight_file(arrFold_trn[:,1:], arrFold_trn[:,0], filename)
    
    #dataCV[f]['val'] = np.array(train_in.iloc[idxVal])
    #dataCV[f]['trn'] = np.array(train_in.iloc[idxTrn])
    
    # save csvs
    #fileVal = 'working_data/fold'+str(fold)+'_val.csv'
    #dataCV[fold]['val'].to_csv(fileVal)
    #fileTrn = 'working_data/fold'+str(fold)+'_trn.csv'
    #dataCV[fold]['trn'].to_csv(fileTrn)
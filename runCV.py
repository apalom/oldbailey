# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:13:26 2020

@author: Alex
"""


#%% cross-validation

def buildCVdata(n_folds, train_in, test_in):
    '''
    build n folds dataset for cross validation

    Parameters
    ----------
    n_folds : int
        number of folds.
    train_in : df
        complete input training dataset.    
    test_in : int
        complete input testing dataset.    

    Returns
    -------
    dataCV : TYPE
        training and validation data folds for cross-validation.

    '''    
    
    # create folder
    date_folder = os.path.join(os.getcwd(), 'working_data',
                               datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(date_folder)
    
    # cross-validation
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
        dataCV[f]['val'] = pd.DataFrame(train_arr[idxVal])
        dataCV[f]['val'] = dataCV[f]['val'].rename(columns={0: 'Label'})
        dataCV[f]['trn'] = pd.DataFrame(train_arr[idxTrn])
        dataCV[f]['trn'] = dataCV[f]['trn'].rename(columns={0: 'Label'})
        
        #arrFold_val = train_arr[idxVal]
        #arrFold_trn = train_arr[idxTrn]    
        
        #filename = os.path.join(date_folder,'fold'+str(f)+'_val.dat')
        #dump_svmlight_file(arrFold_val[:,1:], arrFold_val[:,0], filename)
        #filename = os.path.join(date_folder,'fold'+str(f)+'_trn.dat')
        #dump_svmlight_file(arrFold_trn[:,1:], arrFold_trn[:,0], filename)
    
    return dataCV

dataCV = buildCVdata(5, train_in, test_in)


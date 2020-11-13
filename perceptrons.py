# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:20:36 2020

@author: Alex
"""

# import libraries
import pandas as pd
import numpy as np

#%% calculate error

def pred_acc(data,w,b):
    wT = w.transpose();
    yi_p = []; acc_cnt = 0;
    
    for ix, row in data.iterrows():
        yi = row.Label # select sample label
        xi = row.drop('Label') # select sample features
        
        if np.dot(wT,xi) + b >= 0: # create predicted label
            yi_p.append(1) # true label
        else: yi_p.append(-1) # false label #NOTE check label true/false [1,0] or [1,-1]
    
        if yi_p[-1] == yi:
            acc_cnt += 1; # count correct labels
           
    acc = acc_cnt/len(data)                 
    
    return acc

#%% numpy based avg perceptron
def avgPerc_np(data,w,b,r,T):
    data_np = data.to_numpy()
    y = data_np[:,0]
    X = data_np[:,1:]
    
    w_sum = np.copy(w); b_sum = np.copy(b); s = 0;# initialize values
    acc0 = 0 # initialize accuracy baseline
    lc = np.zeros((T)) # learning curve
    up = 0 # initialize update count
    idx = np.arange(X.shape[0]) # index for stepping through data
    
    for ep in range(T):
        np.random.shuffle(idx1) # shuffle index
    
        for i in idx:
            yi = y[i]
            xi = X[i]      
            
            if yi * (np.dot(w.T,xi) + b) <= 0: # mistake LTU
                w += r * yi * xi; # update weight matrix                 
                b += r * yi; # update bias term
                up += 1;                
            
            # accumulate weights        
            w_sum += w; 
            b_sum += b;
            s += 1;
        
        w_avg = w_sum/s; # average weight
        b_avg = b_sum/s; # average bias 
        
        #make label predictions
        #epAcc = pred_acc(data,w_avg,b_avg) 
        
        lc[ep] = epAcc
            
        # update results if accuracy improves
        if epAcc > acc0: 
            #bestEp = [ep, epAcc, up];
            w_best = w_avg; 
            b_best = b_avg;
            acc0 = epAcc;
        
        print('-> {:.3f}'.format(epAcc), end=" ")
    
    return w_best, b_best, lc

#%% pandas based avg perceptron
def avgPerc_pd(data,w,b,r,T):   
       
    wT = w.transpose(); w_sum = w; b_sum = b; s = 0;# initialize values
    acc0 = 0 # initialize accuracy baseline
    lc = np.zeros((T)) # learning curve
    up = 0 # initialize update count
    for ep in range(T):   

        data = data.sample(frac=1).reset_index(drop=True) # shuffle data
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features
                    
            if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                wT = w.transpose()
                up += 1;                
            
            # accumulate weights
            w_sum += w; 
            b_sum += b;
            s += 1;
        
        w_avg = w_sum/s; # average weight
        b_avg = b_sum/s; # average bias    
        
        # store best accuracy from epochs
        epAcc = pred_acc(data,w_avg,b_avg) 
        
        lc[ep] = epAcc
        
        # update results if accuracy improves
        if epAcc > acc0: 
            #bestEp = [ep, epAcc, up];
            w_best = w_avg; 
            b_best = b_avg;
            acc0 = epAcc;
        
        print('-> {:.3f}'.format(epAcc), end=" ")
    
    return w_best, b_best, lc

#%% standard batch perceptron algorithm
def perc_std(data,w,b,r,T):      
    
    #print('\nBatch Perceptron - Standard:')  
    
    wT = w.transpose(); # initialize values
    acc0 = 0 # initialize accuracy baseline
    lc = np.zeros((T)) # learning curve
    up = 0 # initialize update count
    for ep in range(T):   
        #print('.', end=" ")
        data = data.sample(frac=1).reset_index(drop=True)
        
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features           
            
            if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                up += 1
                wT = w.transpose()
                
        # store best accuracy from epochs        
        epAcc = pred_acc('Standard',data,w,b) 
        lc[ep] = epAcc;
        if epAcc > acc0: 
            #print([ep, np.round(epAcc,4), up])
            best_epAcc = [ep, epAcc, up];
            w_best = w; b_best = b;
            acc0 = epAcc;
    
    #print('\n- Learning rate:', r)    
    #print('\nBatch Perceptron - Standard:', r) 
    return w, b, best_epAcc, lc, w_best, b_best
        
#%% batch perceptron algorithm with learning decay      
def perc_decay(data,w,b,r,T):   
   
    #print('\nBatch Perceptron - Learning Decay')   
    
    wT = w.transpose(); t = 0; r0 = r; # initialize values
    acc0 = 0 # initialize accuracy baseline
    lc = np.zeros((T)) # learning curve
    up = 0 # initialize update count
    for ep in range(T):   
        #print('.', end=" ")
        data = data.sample(frac=1).reset_index(drop=True)
        t += 1; # update t every Epoch (improvement after Oct. 1 class)
        
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features            
            
            if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                up += 1;
                wT = w.transpose()
                
            #t += 1; # update time step
            r = r0/(1+t); # decay learning rate            
        
        # store best accuracy from epochs
        epAcc = pred_acc('Decay',data,w,b) 
        lc[ep] = epAcc;
        if epAcc > acc0: 
            best_epAcc = [ep, epAcc, up];
            w_best = w; b_best = b;
            acc0 = epAcc;
        
    #print('\n- Learning rate:', r0, '->', np.round(r,6))
    #print('\nBatch Perceptron - Learning Decay:', r0, '->', np.round(r,6))   
    return w, b, best_epAcc, lc, w_best, b_best

#%% batch perceptron algorithm with margin and decaying learning rate  
def perc_margin(data,w,b,r,T,margin):   
   
    #print('\nBatch Perceptron - Margin + Decay')   
    
    wT = w.transpose(); t = 0; r0 = r; # initialize values
    acc0 = 0 # initialize accuracy baseline
    lc = np.zeros((T)) # learning curve
    up = 0 # initialize update count
    for ep in range(T):   
        #print('.', end=" ")
        data = data.sample(frac=1).reset_index(drop=True)
        
        for ix, row in data.iterrows():
            yi = row.Label # select sample label
            xi = row.drop('Label') # select sample features
            
            if yi * (np.dot(wT,xi) + b) <= margin: # mistake LTU
                w += r * yi * xi # update weight matrix
                b += r * yi # update bias term
                up += 1;
                wT = w.transpose()
                
            t += 1; # update time step
            r = r0/(1+t); # decay learning rate            
    
        # store best accuracy from each epoch
        epAcc = pred_acc('Margin',data,w,b) 
        lc[ep] = epAcc;
        if epAcc > acc0: 
            best_epAcc = [ep, epAcc, up];
            w_best = w; b_best = b;
            acc0 = epAcc;    
    
    #print('\nBatch Perceptron - Margin + Decay', r, margin)   
    #print('\n- Learning rate:', r0, '->', np.round(r,6))
    #print('- Margin:', margin)
    return w, b, best_epAcc, lc, w_best, b_best

#%% run

# get data
# data, X, y = load_trainData('data/csv-format/train.csv')

# # initialize parameters
# np.random.seed(42) # set random seed
# etas = [1, 0.1, 0.01] # learning rates 
# margins = [1, 0.1, 0.01] # margins 

# # initialize weights and bias terms
# predAcc_dict = {}; # store accuracies
# w0 = np.random.uniform(-0.01, 0.01, size=(X.shape[1])) 
# b0 = np.random.uniform(-0.01, 0.01)
# T = 10;

# w_std, b_std = perc_std(data,w0,b0,etas[0],T)
# predAcc_dict['std'] = pred_acc('Standard',data,w_std,b_std) 

# w_decay, b_decay = perc_decay(data,w0,b0,etas[0],T)  
# predAcc_dict['decay'] = pred_acc('Learning Decay',data,w_decay,b_decay)   

# w_avg, b_avg = perc_avg(data,w0,b0,etas[0],T)
# predAcc_dict['avg'] = pred_acc('Averaging',data,w_avg,b_avg) 

# w_margin, b_margin = perc_margin(data,w0,b0,etas[0],T,margins[0])   
# predAcc_dict['margin'] = pred_acc('Margin + Decay',data,w_margin,b_margin)  

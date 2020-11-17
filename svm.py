# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:22:24 2020

@author: Alex
"""


#%% develop SVM

def avgSvm(data, g0, C, T):

    data_np = data.to_numpy()
    y = data_np[:,0]
    X = data_np[:,1:]
    
    w = np.zeros(X.shape[1]); b = 0.0;
    acc0 = 0 # initialize accuracy baseline
    lc = np.zeros((T)) # learning curve
    w_sum = np.copy(w); b_sum = np.copy(b); # initialize values
    up = 0; s = 0; # initialize update counters
    idx = np.arange(X.shape[0]) # index for stepping through data
    
    for ep in range(T):
        np.random.shuffle(idx) # shuffle index
        gt = g0/(1+ep)
    
        for i in idx:
            yi = y[i]; xi = X[i];    
        
            if yi*(np.dot(w.T, xi) + b) <= 1:
                w = (1-gt)*w + gt*C*yi*xi;  
                b = (1-gt)*b + gt*C*yi;
                up+=1            
                  
            else: 
                w = (1-gt)*w; 
                b = (1-gt)*b;
            
            s += 1;    
            w_sum += w;
            b_sum += b;
        
        w_avg = w_sum/s;
        b_avg = b_sum/s;
        #make label predictions & return training accuracy                
        epAcc = accuracy(data,w_avg,b_avg)         
        lc[ep] = epAcc #learning curve
            
        # update results if accuracy improves
        if epAcc > acc0: 
            w_best = w_avg; 
            b_best = b_avg;
            acc0 = epAcc;
                      
        print('-> {:.4f}'.format(epAcc), end=" ")
    
    return w_best, b_best, lc



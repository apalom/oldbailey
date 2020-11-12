# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:56:28 2020

@author: Alex
"""


def run_perceptron(dataCV):
    '''    

    Returns
    -------
    None.

    '''
            
    acc_a0 = 0; # initialize baseline accuracy value
    up_a0 = 0; # initialize update counts 
    lc = {}
        
    n_folds = len(dataCV)
    # --- average
    print('<<< Averaging Perceptron >>>')
    for f in np.arange(0, n_folds):
        up_a0 = 0; # initialize updates counter for each fold
        print('\nFold - ', f)
        
        data_fold = dataCV[f]
        
        # initialize parameters
        #etas = [1,0.5,0.1] # learning rates 
        etas = [10] # learning rates 
        
        # initialize weights and bias terms
        #prog = {}; # store accuracies
        w0 = np.random.uniform(-0.01, 0.01, size=(data_fold['trn'].shape[1]-1)) 
        b0 = np.random.uniform(-0.01, 0.01)
        T = 15; # number of epochs
    
        for r in etas:
            print('- Learning rate:', r)
            # perc_avg: averging perceptron model
            #prog[f,r], lc[f,r], w_avg, b_avg = perc_avg(data_fold['trn'],w0,b0,r,T)
            w_avg, b_avg, lc[f] = perc_avg(data_fold['trn'],w0,b0,r,T)
            #trnAcc_a = pred_acc('Avg - Training', data_fold['trn'], w_avg, b_avg) 
            print(' validation accuracy')
            valAcc_a = pred_acc('Avg - Validation', data_fold['val'], w_avg, b_avg) 
            
            if valAcc_a > acc_a0: # update best values if validation accuracy improves                
                #print('  -Update:', up_a0, '|', np.round(acc_a0,3), '->', np.round(valAcc_a,3), f, r, 0)
                acc_a0 = valAcc_a; # if predicted accuracy for these hp is better, update
                up_a0 += 1;            
                # update best values based on validation accuracy
                best_w = w_avg; best_b = b_avg;  best_r = r;
                best_acc = valAcc_a;
                
    return best_w, best_b, best_r, best_acc, lc

# Using current time 
t_st = time.time()
best_w, best_b, best_r, best_acc, lc = run_perceptron(dataCV)
t_en = time.time()
print('Runtime (m):', np.round((t_en - t_st)/60,2))

#%%

from multiprocessing import Pool

if __name__ == '__main__':
    p = Pool(8)
    print(p.map(run_perceptron, ))
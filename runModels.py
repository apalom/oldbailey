# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:56:28 2020

@author: Alex
"""

#%% execute averaging perceptron
def run_perceptron(dataCV):
    '''    
    run average perceptron
    '''
            
    acc_a0 = 0; # initialize baseline accuracy value
    up_a0 = 0; # initialize update counts 
    lc = {}
        
    n_folds = len(dataCV)

    for f in np.arange(0, n_folds):
        up_a0 = 0; # initialize updates counter for each fold
        print('\nFold -', f)
        
        data_fold = dataCV[f]

        T = 25; # number of epochs
        r = 10; margin = 1; g0 = 0.001; C = 10; # hypers
        print('- Learning rate:', r)
        #print('- Gamma0: {} | Tradeoff: {}'.format(g0,C))        
        #w_avg, b_avg, lc[f] = avgPerc_np(data_fold['trn'],r,T)          
        #w_avg, b_avg, lc[f] = margPerc(data_fold['trn'], r, margin, T)
        w_avg, b_avg, lc[f] = margAvgPerc(data_fold['trn'], r, margin, T)
        #w_avg, b_avg, lc[f] = avgSvm(data_fold['trn'], g0, C, T)
        
        # validation accuracy
        valAcc_a = pred_acc(data_fold['val'], w_avg, b_avg) 
        print('\n ==> val acc {:.4f}'.format(valAcc_a))
        
        if valAcc_a > acc_a0: # update best values if validation accuracy improves                                
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

#%% execute SVM perceptron
def run_SVM(dataCV):
    '''    
    run SVM algorithm
    '''
            
    acc_a0 = 0; # initialize baseline accuracy value
    up_a0 = 0; # initialize update counts 
    lc = {}
        
    n_folds = len(dataCV)
    # --- average
    print('<<< SVM Algorithm >>>')
    for f in np.arange(0, n_folds):
        up_a0 = 0; # initialize updates counter for each fold
        print('\nFold -', f)
        
        data_fold = dataCV[f]
        
        # initialize parameters
        #etas = [1,0.5,0.1] # learning rates 
        etas = [0.1] # learning rates         
        #Cs = [1,0.1] # tradeoff
        C = 1;
        
        # initialize weights and bias terms    
        w0 = np.zeros((data_fold['trn'].shape[1])) 
        T = 10; # number of epochs
    
        for r in etas:
            print('- Learning rate:', r)
            
            w_avg, b_avg, lc[f] = perc_avg(data_fold['trn'],w0,b0,r,T)
            #trnAcc_a = pred_acc('Avg - Training', data_fold['trn'], w_avg, b_avg)             
            valAcc_a = pred_acc('Avg - Validation', data_fold['val'], w_avg, b_avg) 
            print('\n ==> val acc {:.3f}'.format(valAcc_a))
            
            if valAcc_a > acc_a0: # update best values if validation accuracy improves                                
                acc_a0 = valAcc_a; # if predicted accuracy for these hp is better, update
                up_a0 += 1;            
                # update best values based on validation accuracy
                best_w = w_avg; best_b = b_avg;  best_r = r;
                best_acc = valAcc_a;
                
    return best_w, best_b, best_r, best_acc, lc

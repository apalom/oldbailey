# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:00:09 2020

@author: Alex
"""


#%% make predictions

def makePred(data, y, w, b):

    wT = w.transpose();
    yi_p = []; acc_cnt = 0;
    
    for i, row in data.iterrows():
         
        xi = row; # data sample
        
        if np.dot(wT,xi) + b >= 0: # create predicted label
            yi_p.append(1) # true label
        else: yi_p.append(0) # false label #NOTE check label true/false [1,0] or [1,-1]

        if isinstance(y, str) == False: 
            yi = y.Label[i]       
            if yi_p[-1] == yi:
                acc_cnt += 1; # count correct labels
        
    acc = acc_cnt/len(data)

    return yi_p, acc

n_meta = len(best_w) - (X_trnComb.shape[1]) # number of meta features
# use best w and b for evaluation
if n_meta > 0:
    w_eval = best_w[:-n_meta] #remove meta data training weights
else: w_eval = best_w
b_eval = best_b;

# combine bow and glove eval datasets
#eval_in = X_bowEval[sig.index]
eval_in = pd.concat([X_bowEval, X_gloEval, X_tfEval], axis=1, sort=False)
eval_in.columns= np.arange(0,eval_in.shape[1])
#eval_in = eval_in[sig.index]

# combine bow and glove eval datasets
#test_in = X_bowTst[sig.index]
test_in = pd.concat([X_bowTst, X_gloTst, X_tfTst], axis=1, sort=False)
test_in.columns= np.arange(0,test_in.shape[1])
#test_in = test_in[sig.index]

pred_Test, acc = makePred(test_in, y_bowTst, w_eval, b_eval)
#print("Avg. test prediction:",np.round(np.mean(pred_Test),4))
print('Test accuracy:', np.round(acc,4))

pred_Eval, _ = makePred(eval_in, 'none', w_eval, b_eval)
#print("Avg. eval prediction:",np.round(np.mean(pred_Eval),4))

pred_Eval = pd.DataFrame(pred_Eval, columns=['label'])
pred_Eval['example_id'] = np.arange(0,len(pred_Eval))
#%%
print('---- save prediction ----')
pred_Eval.to_csv('predictions/yi_p17.csv')

#%%



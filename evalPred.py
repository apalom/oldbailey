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

n_meta = len(best_w) - len(sig) # number of meta features
# use best w and b for evaluation
w_eval = best_w[:-n_meta] #remove meta data training weights
b_eval = best_b;

eval_in = X_bowEval[sig.index]
test_pred = X_bowTst[sig.index]

pred_Test, acc = makePred(test_pred, y_bowTst, w_eval, b_eval)
#print("Avg. test prediction:",np.round(np.mean(pred_Test),4))
print('Test accuracy:', np.round(acc,4))

pred_Eval, _ = makePred(eval_in, 'none', w_eval, b_eval)
#print("Avg. eval prediction:",np.round(np.mean(pred_Eval),4))

pred_Eval = pd.DataFrame(pred_Eval, columns=['label'])
pred_Eval['example_id'] = np.arange(0,len(pred_Eval))

pred_Eval.to_csv('yi_p.csv')

#%%



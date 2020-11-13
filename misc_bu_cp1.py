# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:59:41 2020

@author: Alex
"""


#%%
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

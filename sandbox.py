# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:15:32 2020

@author: Alex
"""
import numpy as np

np.random.seed(42)
data = dataCV[0]['trn']

dataDummy = np.array([[-1,1,0,0,1],
                      [-1,1,0,0,1],
                      [1,0,0,1,0],
                      [1,0,1,1,0],
                      [1,0,1,1,0]])

data = pd.DataFrame(dataDummy,columns=['Label','A','B','C','D'])

T=5; r=1;

data_np = data.to_numpy()
y = data_np[:,0]
X = data_np[:,1:]

w = np.random.uniform(-0.01, 0.01, size=(X.shape[1]))
b = np.random.uniform(-0.01, 0.01)
#y = data.Label.to_numpy()
#X = data.drop(columns=['Label']).to_numpy()

#w_sum = np.copy(w); b_sum = np.copy(b); # initialize values
w_sum = w; b_sum = b; # initialize values
wT = w.transpose();
acc0 = 0 # initialize accuracy baseline
lc = np.zeros((T)) # learning curve
up = 0; s = 0; # initialize update count
idx = np.arange(X.shape[0])   

# sample run perceptron
for ep in range(T):   
    
    np.random.shuffle(idx1)
    for i in idx:
        yi = y[i]
        xi = X[i]    

        if yi * (np.dot(wT,xi) + b) <= 0: # mistake LTU
            w += r * yi * xi; # update weight matrix                 
            b += r * yi; # update bias term
            wT = w.transpose();
            up += 1;                
        
        # accumulate weights        
        w_sum += w; 
        b_sum += b;
        s += 1;
    
    w_avg = w_sum/s; # average weight
    b_avg = b_sum/s; # average bias    
    
    # store best accuracy from epochs
    #epAcc = pred_acc(data,w_avg,b_avg) 
    
    #make label predictions   
    w_avgT = w_avg.transpose()
    yi_p = []; acc_cnt = 0;
    for i in range(X.shape[0]):
        yi = y[i]
        xi = X[i]
        if (np.dot(w_avgT,xi) + b_avg) >= 0: # create predicted label
            yi_p.append(1) # true label
        else: yi_p.append(-1) # false label #NOTE check label true/false [1,0] or [1,-1]
    
        if yi_p[-1] == int(yi):
            acc_cnt += 1; # count correct labels
       
    epAcc = acc_cnt/X.shape[0]   
    
    lc[ep] = epAcc
        
    # update results if accuracy improves
    if epAcc > acc0: 
        #bestEp = [ep, epAcc, up];
        w_best = w_avg; 
        b_best = b_avg;
        acc0 = epAcc;
    
    print('-> {:.3f}'.format(epAcc), end=" ")
    
#%%

w = np.random.uniform(-0.01, 0.01, size=(data.shape[1]-1)) 
b = np.random.uniform(-0.01, 0.01)

wT = w.transpose(); w_sum = w; b_sum = b; s = 0;# initialize values
acc0 = 0 # initialize accuracy baseline
lc = np.zeros((T)) # learning curve
up = 0 # initialize update count
#%%
for ep in range(T):   

    data = data.sample(frac=1).reset_index(drop=True) # shuffle data
    for ix, row in data.iterrows():
        yi = row.Label # select sample label
        xi = row.drop('Label').to_numpy() # select sample features    
        
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
    
    #print('.', end=" ")        
    #print('val acc: {:.4f}'.format(ep,acc), end=" ")
    print('-> {:.3f}'.format(epAcc), end=" ")
    
#%%

  
#%% Using current time 
d = dataCV[0]['trn']
data_np = d.to_numpy()
y = data_np[:,0]
X = data_np[:,1:]

g0 = 0.01; C = 10000; T = 10;

t_st = time.time()
w_best, b_best, lc = svm(d,g0,C,T)
t_en = time.time()
print('\nRuntime (m):', np.round((t_en - t_st)/60,3))

#%%
g0 = 1; C = 0.1;
d = dataCV[0]['trn']

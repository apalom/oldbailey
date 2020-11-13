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


#%%

for i in range(X.shape[0]):
    
    val = y[i]*np.dot(w.T,X.iloc[i])
    print(val)
    if val > 0: # create predicted label
        yi_p.append(1) # true label
    else: yi_p.append(-1) # false label #NOTE check label true/false [1,0] or [1,-1]

    if yi_p[-1] == y[i]:
        acc_cnt += 1; # count correct labels
    acc = acc_cnt/len(X)    
    
print(acc)

#%%

X = np.array([[0,0,0],
              [0,1,0],
              [1,0,0],
              [1,1,0]])

y = np.array([[-1],[-1],[-1],[1]])

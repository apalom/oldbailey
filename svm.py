# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:22:24 2020

@author: Alex
"""


#%% develop SVM

def svm(data, g0, C, T):

    y = d.Label.values
    X = d.drop(columns=['Label']).values
    w = np.zeros((X.shape[1])); b = 0;
    w_sum = w; b_sum = b; s = 0; up = 0;# initialize values 
    
    for ep in range(T):
        np.random.shuffle(X)
        gt = g0/(1+ep)
        
        for i in range(X.shape[0]): # update weights
        
            if y[i]*(np.dot(w.T, X[i]) + b) <= 1:
                w = (1-gt)*w + gt*C*y[i]*X[i]     
                b = (1-gt)*b + gt*C*y[i]
                up+=1
                  
            else: 
                w = (1-gt)*w; 
                b = (1-gt)*b;
                
        yi_p = []; acc_cnt = 0;
        
        for i in range(X.shape[0]):
            
            #if y[i]*np.dot(wT,X[i]) >= 1: # create predicted label
            if np.dot(w.T,X[i]) + b >= 0: # create predicted label
                yi_p.append(1) # true label
            else: yi_p.append(-1) # false label #NOTE check label true/false [1,0] or [1,-1]
        
            if yi_p[-1] == y[i]:
                acc_cnt += 1; # count correct labels
               
        acc = acc_cnt/len(X)                 
        print('-> {:.3f}'.format(acc), end=" ")
    
    return w, b
  
#%% Using current time 
t_st = time.time()
g0 = 0.01; C = 10000;
w, b = svm(d,g0,C,10)
t_en = time.time()
print('\nRuntime (m):', np.round((t_en - t_st)/60,3))

#%%
g0 = 1; C = 0.1;
d = dataCV[0]['trn']
y = d.Label              
X = d.drop(columns=['Label'])

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

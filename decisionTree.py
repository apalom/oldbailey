# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:20:09 2020

@author: Alex
"""

# import libraries
import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

#%% load training data
print('Implementation - Full Trees with Cross Validation')

'''
Using Adult data set from UCI ML repo (file a1a.train). 14 features: 6 continuous,
8 categorical. Continuous features are discretized. Categorical features with 
m categories are converted to m binary features. 
'''

# Using readlines() 
def readData():
    D = {}; 
    for fold in range(1,6):
        D[fold] = {}
        
        file1 = open('data/CVfolds/fold'+str(fold), 'r') 
        Lines = file1.readlines() 
        
        maxIx = 119;
        X = pd.DataFrame(np.zeros((len(Lines),maxIx+1)))
        
        # parse libsvm
        r = 0; m = 0;
        for line in Lines: 
            X.iloc[r][0] = int(line[:2]) # instance label     
            features = line[3:-2].split() # instance features
            
            for d in features:
                d1 = d.split(':')
                ix = int(d1[0])
                X.iloc[r][ix+1] = 1        
        
            r += 1
            #print('fold:',fold, '| row >', r)
            
        # define training data
        X = X.rename(columns={0: 'Label'})
        # define attributes
        A = list(X.columns)
        # define training labels
        y = X.Label
        S = X
        
        D[fold]['X'] = X
        D[fold]['y'] = y
        D[fold]['S'] = S
        D[fold]['A'] = A
            
    return D

D = readData()

#%% label probabilities

def labelP(y):
    labels = list(set(list(y.values))); p = [];
    for lbl in labels:
        p.append(list(y.values).count(lbl)/len(y)) # probabilty of Yes label
    return p

# define entropy
def entropy(p):    
    H = 0    
    for p_lbl in p:                
        H += -p_lbl * math.log2(p_lbl)
        
    return(H)

#%% define information gain

def infoGain(S,A):
    A_prob = [] # attribute probabilities
    A_names = {}; # attribute names   
    y = S['Label']; # get data labels
    H = entropy(labelP(y)) # calculate entropy
    ss_v = [];
    for a in A:        
        V = list(set(S[a].values)) # possible values in attribute
        S_a = S[a] # feature column    
        a0 = []; s0 = [];
        for v in V:
            Sv = S[S_a==v] # return subset values             
            y_av = Sv['Label'] 
            y_av_lbl = list(set(y_av))
            s0.append(len(y_av)); # len of subset
            a1 = [];
            for lbl in y_av_lbl:            
                a1.append(list(y_av.values).count(lbl)/(len(y_av))) # probabilty of Yes label
                A_names[a,v,lbl] = [a1, len(y_av)]     
                            
            a0.append(list(a1))
        
        ss_v.append(s0) # subset probabilities
        A_prob.append(a0) # attribute probabilities                                   
        
    i = 0; A_all = {}; G_all = {};    
    for att in A_prob:        
        I = 0;     
        for k in range(len(att)):                          
            I += (ss_v[i][k]/len(S)) * entropy(att[k]) #id3 update                       
        
        A_all[A[i]] = A_prob[i] # store all A probs
        G_all[A[i]] = (H - I) # information gain      
        i += 1;
            
    return G_all, A_all, A_prob

#%% id3 build tree

'''
Implement the decision tree data structure and the ID3 algorithm for
your decision tree.
https://stackoverflow.com/questions/11479624/is-there-a-way-to-guarantee-hierarchical-output-from-networkx
'''

# initialize tree structure
print('--- Training classifier ---')
G = nx.DiGraph(); tree = {};
def id3(S,A,b):
    
    # if all examples have the same label return a single node tree
    lbls = list(set(S.Label))
    if len(lbls) == 1: 
        G.add_node(lbls[0]) 
        tree['single'] = lbls[0]
            
    # otherwise we need to build an actual tree
    else:
        # return information gain among attributes in A and add 'Label' for InfoGain function        
        Gain, _, _ = infoGain(S,A) 
        best = max(Gain,key=Gain.get) # identify best attribute

        # add subtree
        if b != 'none':
            tree[b[0]][b[1]] = best
            G.add_edge(b[0],best)
            #print('Add \|/', b[0],'[', b[1],']')
        
        # add root node
        G.add_node(best)        
        #print('Root:', best)
        tree[best] = {}
        
        # for each possible value v in attribute
        v_in_A = list(set(S[best]))
        for v in v_in_A:          
            # create subset of examples in S with A = v
            Sv = S[S[best] == v]
            Sv_lbl = list(set(Sv.Label))
            #print('  Add /', best ,'[', v, ']')
            tree[best][v] = {}
            
            # if empty subset, then add most common label
            if len(Sv_lbl) == 0:  
                maj_lbl, _ = Counter(Sv['Label']).most_common()[0] # return majority label in Sv
                maj_lbl0 = str(v)+' = '+str(maj_lbl)
                #print('  Add <>', maj_lbl,'- for empty subset of [', best,']')
                G.add_node(maj_lbl0)
                G.add_edge(best,maj_lbl0)
                tree[best][v] = maj_lbl                
            
            # add leaf node
            elif len(Sv_lbl) == 1:    
                leaf = Sv['Label'].values[0]
                leaf0 =  str(v)+' = '+str(leaf)
                #print('  Add <>', best , '[', v, '] =', leaf)                
                G.add_node(leaf0)
                G.add_edge(best,leaf0)        
                tree[best][v] = leaf                
            
            # else build subtree with S - previous best attribute
            else:
                # remove used attribute from list and update     
                Anew = A.copy()
                Anew.remove(best)                                
                
                # branch connecting current root to next root                 
                b = [best,v]    
                # recurse subtree
                #print('--->')
                id3(Sv, Anew, b)
    
    root0 = max(Gain,key=Gain.get);
    #print('Root node:', root0)
    #print('Root node information gain: {:.3f}'.format(Gain[root0]))
    
    return G, tree

#%% data evaluation

def dt_class(X,root0,tree):
    lbl_tree = {}; accurate = 0; maxDepth = 0;
    # traverse tree classifier
    for idx, x in X.iterrows():
        
        next_leaf = root0; depth = 0;
        while type(next_leaf) != np.float64:        
            split_val = x[next_leaf]   
            next_leaf = tree[next_leaf][split_val]
            depth += 1
            
            if depth >= 3:
                next_leaf = split_val
            
        lbl_tree[idx] = next_leaf
        lbl_X = X.Label.loc[idx]
        accurate += 1 if next_leaf == lbl_X else 0
        maxDepth = max(maxDepth,depth)
    
    err = 1-accurate/len(X)
    print('Classification error = {:.3f}'.format(err))  
    print('Max tree depth = {}'.format(maxDepth))  

    return lbl_X, err, maxDepth
    
#%% cross-validation
'''
For this problem we will implement k-folds cross-validation.
'''

results = {}
for fold in range(1,6):

    results[fold] = {}
    
    print('\n+++ Data Collection -', fold,' +++')
    tst_X = D[fold]['X'] 
    tst_y = D[fold]['y'] 
    tst_S = D[fold]['S']     
    A = D[fold]['A']
    
    # create list of all other folds for training        
    other_folds = list(range(1,6))  
    del other_folds[fold-1]
    
    trn_X = D[other_folds[0]]['X']     
    trn_y = D[other_folds[0]]['y']   
    trn_S = D[other_folds[0]]['S']       
    del other_folds[0]
    
    for f in other_folds:
        trn_X = trn_X.append(D[f]['X'], ignore_index=True)
        trn_y = trn_y.append(D[f]['y'], ignore_index=True)
        trn_S = trn_S.append(D[f]['S'], ignore_index=True)
        
    p = labelP(trn_y)
    print('Label probabilities:', p)
    print('Entropy: {:.4f}'.format(entropy(p)))

    print('+++ Calculate Gain -', fold,' +++')
    Gain, A_all, A_prob = infoGain(trn_S,A)
    del Gain['Label']
    #Gain = Gain.copy()
    #del Gain_X['Label']
           
    print('+++ Build Tree -', fold,' +++')
    A = list(trn_S.columns)[1:]
    G, tree = id3(trn_S,A,'none')
    
    #print('\n',tree)
    root0 = max(Gain,key=Gain.get);
    print('Root node:', root0)
    print('Root node information gain: {:.3f}'.format(Gain[root0]))
    
    print('+++ Training -', fold,' +++')
    lbl_Xtrn, train_err, maxDepth = dt_class(trn_X,root0,tree)
    print('\n-- Testing -', fold,' +++')
    lbl_Xtest, test_err, _ = dt_class(tst_X,root0,tree)

    results[fold]['entropy'] = entropy(p)
    results[fold]['root'] = root0
    results[fold]['root_I'] = Gain[root0]
    results[fold]['maxDepth'] = maxDepth   
    results[fold]['train_err'] = train_err
    results[fold]['test_err'] = test_err

#print('\nMax depth = 3')
#print(results)

# order root value by entropy gain
# roots = pd.DataFrame.from_dict(Gain,orient='index')
# roots = roots[0].rename('Entropy')

# roots = roots.sort_values(ascending = False)
# root0 = roots.index[0]

















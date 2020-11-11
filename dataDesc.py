# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:14:46 2020

@author: Alex
"""

#%% check meta categories len and distribution

featureize = {}
for f in list(metaTrn):
    featureize[f] = metaTrn[f].value_counts()
    
    featureize[f+'_count'] = len(set(metaTrn[f]))
    featureize[f+'_set'] = list(set(metaTrn[f]))

#%% describe meta training data
metaTrn.describe()
metaTrn_totals = np.sum(metaTrn)

crimes = ['Damage', 'Deception', 'Kill', 'Misc', 'Royal Offense', 'Sexual', 'Theft', 'Violent Theft']
metaTrn_crimes = metaTrn_totals[-8:]
metaTrn_crimes.index = crimes

#%% plot parameters

cm = plt.cm.get_cmap('Reds')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 14

#%% descriptive statistics - defendant age
fig, ax = plt.subplots(1, 1, figsize=(10,6), dpi=100)

g0 = sns.distplot(metaTrn['defendant_age'].where(metaTrn['defendant_age'] > 0),
    bins = np.arange(0,maxAge,5), label='Known Ages')
g1 = sns.distplot(metaTrn['defendant_age'].where(metaTrn['defendant_age'] == 0),
    bins = np.arange(0,maxAge,5), label='Unknown Ages')

plt.title('Defendant Age Distribution')
maxAge = roundup(np.max(metaTrn['defendant_age']),10)
plt.xlim([-0.5,maxAge])
plt.legend()

#sns.distplot(metaTrn['defendant_age'].where(metaTrn['defendant_age'] > 0).dropna(),
#    bins = np.arange(0,maxAge,5), label='Known Ages')
#sns.distplot(metaTrn['defendant_age'].where(metaTrn['defendant_age'] == 0).dropna(),
#    bins = np.arange(0,maxAge,5), label='Unknown Ages')
#metaTrn['defendant_age'].where(metaTrn['defendant_age'] > 0).dropna().plot.hist(
#    bins = np.arange(0,maxAge,10), alpha=0.8, rwidth=0.95, label='Known Ages')
#metaTrn['defendant_age'].where(metaTrn['defendant_age'] == 0).dropna().plot.hist(
#    bins=np.arange(0,20,10), alpha=0.8, rwidth=0.95, label='Unknown Ages')

#sns.distplot(metaTrn['defendant_age'].dropna(),
#    bins = np.arange(0,maxAge,5), label='Known Ages')

#%% descriptive statistics - number of victims 

fig, ax = plt.subplots(1, 1, figsize=(10,6), dpi=100)
maxVictimBnd = np.round(np.mean(metaTrn['num_victims'])) + 3*np.round(np.std(metaTrn['num_victims'])) + 1

metaTrn['num_victims'].plot.hist(density=True, bins=np.arange(0,maxVictimBnd,1))

plt.xticks(np.arange(0,maxVictimBnd))
plt.title('Number of Victims Distribution')

#%% descriptive statistics - crime distribution

fig, ax = plt.subplots(1, 1, figsize=(10,6), dpi=100)

metaTrn_crimes.plot.bar()
plt.title('Crime Comparison')
plt.ylabel('Count')
plt.xticks(rotation=45)


#%%

cols = list(train_in)
cols[0] = 'Label'
train_in.columns = cols
cols = list(test_in)
cols[0] = 'Label'
test_in.columns = cols

train_in.Label = train_in['Label'].replace(0,-1)
test_in.Label = test_in['Label'].replace(0,-1)


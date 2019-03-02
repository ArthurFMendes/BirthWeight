#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:16:27 2019

@author: arthurmendes

Working Directory:
/Users/arthurmendes/Desktop/BirthData

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


file = 'birthweight.xlsx'


birthdata = pd.read_excel(file)


########################
# Fundamental Dataset Exploration
########################

# Column names
birthdata.columns


# Displaying the first rows of the DataFrame
print(birthdata.head())


# Dimensions of the DataFrame
birthdata.shape


# Information about each variable
birthdata.info()


# Descriptive statistics
birthdata.describe().round(2)


birthdata.sort_values('bwght', ascending = False)

###############################################################################
# Imputing Missing Values
###############################################################################

print(
      birthdata
      .isnull()
      .sum()
      )


for col in birthdata:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if birthdata[col].isnull().any():
        birthdata['m_'+col] = birthdata[col].isnull().astype(int)
        
        
# Creating a dropped dataset to graph 'cigs' & 'drink'
df_dropped = birthdata.dropna()

sns.distplot(df_dropped['meduc'])

sns.distplot(df_dropped['monpre'])

sns.distplot(df_dropped['npvis'])

sns.distplot(df_dropped['fage'])

sns.distplot(df_dropped['feduc'])

sns.distplot(df_dropped['omaps'])

sns.distplot(df_dropped['fmaps'])

sns.distplot(df_dropped['cigs'])

sns.distplot(df_dropped['drink'])



# # Filled with the median

fill = birthdata['meduc'].median()

birthdata['meduc'] = birthdata['meduc'].fillna(fill)


fill = birthdata['monpre'].median()

birthdata['monpre'] = birthdata['monpre'].fillna(fill)


fill = birthdata['npvis'].median()

birthdata['npvis'] = birthdata['npvis'].fillna(fill)


fill = birthdata['fage'].median()

birthdata['fage'] = birthdata['fage'].fillna(fill)


fill = birthdata['feduc'].median()

birthdata['feduc'] = birthdata['feduc'].fillna(fill)


fill = birthdata['omaps'].median()

birthdata['omaps'] = birthdata['omaps'].fillna(fill)


fill = birthdata['fmaps'].median()

birthdata['fmaps'] = birthdata['fmaps'].fillna(fill)


fill = birthdata['cigs'].median()

birthdata['cigs'] = birthdata['cigs'].fillna(fill)


fill = birthdata['drink'].median()

birthdata['drink'] = birthdata['drink'].fillna(fill)

# Checking the overall dataset to see if there are any remaining
# missing values
print(
      birthdata
      .isnull()
      .any()
      .any()
      )
# Cigs per week 


for val in enumerate(birthdata.loc[ : , 'cigs']):
    
    birthdata.loc[val[0], 'cigs'] = birthdata.loc[val[0], 'cigs']*7

#################################
# Normilize 
#################################

# Cigs
birthdata['cigs'] = (birthdata['cigs'] - birthdata['cigs'].min())/(
        birthdata['cigs'].max() - birthdata['cigs'].min())

# Drinks
 
birthdata['drink'] = (birthdata['drink'] - birthdata['drink'].min())/(
        birthdata['drink'].max() - birthdata['drink'].min())

# mage
birthdata['mage'] = (birthdata['mage'] - birthdata['mage'].min())/(
        birthdata['mage'].max() - birthdata['mage'].min())

# meduc
birthdata['meduc'] = (birthdata['meduc'] - birthdata['meduc'].min())/(
        birthdata['meduc'].max() - birthdata['meduc'].min())

# monpre
birthdata['monpre'] = (birthdata['monpre'] - birthdata['monpre'].min())/(
        birthdata['monpre'].max() - birthdata['monpre'].min())

# npvis
birthdata['npvis'] = (birthdata['npvis'] - birthdata['npvis'].min())/(
        birthdata['npvis'].max() - birthdata['npvis'].min())

# fage
birthdata['fage'] = (birthdata['fage'] - birthdata['fage'].min())/(
        birthdata['fage'].max() - birthdata['fage'].min())

# feduc
birthdata['feduc'] = (birthdata['feduc'] - birthdata['feduc'].min())/(
        birthdata['feduc'].max() - birthdata['feduc'].min())

# omaps
birthdata['omaps'] = (birthdata['omaps'] - birthdata['omaps'].min())/(
        birthdata['omaps'].max() - birthdata['omaps'].min())

# fmaps
birthdata['fmaps'] = (birthdata['fmaps'] - birthdata['fmaps'].min())/(
        birthdata['fmaps'].max() - birthdata['fmaps'].min())

# bwght
birthdata['bwght'] = (birthdata['bwght'] - birthdata['bwght'].min())/(
        birthdata['bwght'].max() - birthdata['bwght'].min())


###############################################################################
# Outlier Analysis
###############################################################################


birthdata_quantiles = birthdata.loc[:, :].quantile([0.20,
                                                0.40,
                                                0.60,
                                                0.80,
                                                1.00])

    
print(birthdata_quantiles)



"""

Assumed Continuous/Interval Variables - 

mage
meduc
monpre
npvis
fage
feduc
omaps
fmaps
cigs
drink



Binary Classifiers -
male
mwhte
mblck
moth
fwhte
fblck
foth
bwght

m_meduc
m_monpre
m_npvis
m_fage
m_feduc
m_omaps
m_fmaps
m_cigs
m_drink

"""



########################
# Visual EDA (Histograms)
########################


plt.subplot(2, 2, 1)
sns.distplot(birthdata['mage'],
             bins = 35,
             color = 'g')

plt.xlabel('Mother Age')



########################


plt.subplot(2, 2, 2)
sns.distplot(birthdata['meduc'],
             bins = 30,
             color = 'y')

plt.xlabel('Mother Education Years')



########################


plt.subplot(2, 2, 3)
sns.distplot(birthdata['monpre'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('month prenatal care began')



########################


plt.subplot(2, 2, 4)

sns.distplot(birthdata['npvis'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('total number of prenatal visits')



plt.tight_layout()
plt.savefig('Birth Data 1 of 3.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birthdata['fage'],
             bins = 35,
             color = 'g')

plt.xlabel('Father Age')


########################


plt.subplot(2, 2, 2)
sns.distplot(birthdata['feduc'],
             bins = 30,
             color = 'y')

plt.xlabel('Father Education Years')



########################


plt.subplot(2, 2, 3)
sns.distplot(birthdata['omaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('one minute apgar score')



########################


plt.subplot(2, 2, 4)

sns.distplot(birthdata['fmaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('five minute apgar score')



plt.tight_layout()
plt.savefig('Birth Data 2 of 3.png')

plt.show()


########################
########################


plt.subplot(2, 1, 1)
sns.distplot(birthdata['cigs'],
             bins = 35,
             color = 'g')

plt.xlabel('avg cigarettes per day')


########################


plt.subplot(2, 1, 2)
sns.distplot(birthdata['drink'],
             bins = 30,
             color = 'y')

plt.xlabel('avg drinks per week')


plt.tight_layout()
plt.savefig('Birth Data 3 of 3.png')

plt.show()


birthdata_quantiles = birthdata.loc[:, :].quantile([0.05,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])

# Outlier flags
mage_lo = 20
mage_hi = 40


meduc_lo = 10
meduc_hi = 17

monpre_lo = 1
monpre_hi = 5


npvis_lo = 4
npvis_hi = 15


fage_lo = 20
fage_hi = 43


feduc_lo = 10
feduc_hi = 17


omaps_lo = 5
omaps_hi = 10


fmaps_lo = 6
fmaps_hi = 10


cigs_hi = 0

drink_hi = 0


########################
# Creating Outlier Flags
########################

# Building loops for outlier imputation



########################
# mage

birthdata['out_mage'] = 0


for val in enumerate(birthdata.loc[ : , 'mage']):
    
    if val[1] < mage_lo:
        birthdata.loc[val[0], 'out_mage'] = -1


for val in enumerate(birthdata.loc[ : , 'mage']):
    
    if val[1] >= mage_hi:
        birthdata.loc[val[0], 'out_mage'] = 1


########################
# meduc

birthdata['out_meduc'] = 0


for val in enumerate(birthdata.loc[ : , 'meduc']):
    
    if val[1] < meduc_lo:
        birthdata.loc[val[0], 'out_meduc'] = -1


for val in enumerate(birthdata.loc[ : , 'meduc']):
    
    if val[1] > meduc_hi:
        birthdata.loc[val[0], 'out_meduc'] = 1

########################
# monpre

birthdata['out_monpre'] = 0


for val in enumerate(birthdata.loc[ : , 'monpre']):
    
    if val[1] < monpre_lo:
        birthdata.loc[val[0], 'out_monpre'] = -1


for val in enumerate(birthdata.loc[ : , 'monpre']):
    
    if val[1] > monpre_hi:
        birthdata.loc[val[0], 'out_monpre'] = 1
########################
# npvis

birthdata['out_npvis'] = 0


for val in enumerate(birthdata.loc[ : , 'npvis']):
    
    if val[1] < npvis_lo:
        birthdata.loc[val[0], 'out_npvis'] = -1


for val in enumerate(birthdata.loc[ : , 'npvis']):
    
    if val[1] > npvis_hi:
        birthdata.loc[val[0], 'out_npvis'] = 1


########################
# fage

birthdata['out_fage'] = 0


for val in enumerate(birthdata.loc[ : , 'fage']):
    
    if val[1] < fage_lo:
        birthdata.loc[val[0], 'out_fage'] = -1


for val in enumerate(birthdata.loc[ : , 'fage']):
    
    if val[1] > fage_hi:
        birthdata.loc[val[0], 'out_fage'] = 1
########################
# feduc

birthdata['out_feduc'] = 0


for val in enumerate(birthdata.loc[ : , 'feduc']):
    
    if val[1] < feduc_lo:
        birthdata.loc[val[0], 'out_feduc'] = -1


for val in enumerate(birthdata.loc[ : , 'feduc']):
    
    if val[1] > feduc_hi:
        birthdata.loc[val[0], 'out_feduc'] = 1
########################
# omaps

birthdata['out_omaps'] = 0


for val in enumerate(birthdata.loc[ : , 'omaps']):
    
    if val[1] < omaps_lo:
        birthdata.loc[val[0], 'out_omaps'] = -1


for val in enumerate(birthdata.loc[ : , 'omaps']):
    
    if val[1] >= omaps_hi:
        birthdata.loc[val[0], 'out_omaps'] = 1
########################
# fmaps

birthdata['out_fmaps'] = 0


for val in enumerate(birthdata.loc[ : , 'fmaps']):
    
    if val[1] < fmaps_lo:
        birthdata.loc[val[0], 'out_fmaps'] = -1


for val in enumerate(birthdata.loc[ : , 'fmaps']):
    
    if val[1] >= fmaps_hi:
        birthdata.loc[val[0], 'out_fmaps'] = 1
########################
# cigs

birthdata['out_cigs'] = 0

for val in enumerate(birthdata.loc[ : , 'cigs']):
    
    if val[1] > cigs_hi:
        birthdata.loc[val[0], 'out_cigs'] = 1


########################
# drink

birthdata['out_drink'] = 0

for val in enumerate(birthdata.loc[ : , 'drink']):
    
    if val[1] > drink_hi:
        birthdata.loc[val[0], 'out_drink'] = 1


###############################################################################
# Correlation Analysis
###############################################################################

birthdata.head()


df_corr = birthdata.corr().round(2)


print(df_corr)


df_corr.loc['bwght'].sort_values(ascending = False)



########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Birth Data Correlation Heatmap.png')
plt.show()



###############################################################################
# Univariate Regression Analysis
###############################################################################

# Building a Regression Base
lm_price_qual = smf.ols(formula = """bwght ~ birthdata['mage']""",
                         data = birthdata)



# Fitting Results
results = lm_price_qual.fit()


# Printing Summary Statistics
print(results.summary())


'''
Binary Classifiers -
male
mwhte
mblck
moth
fwhte
fblck
foth


m_meduc
m_monpre
m_npvis
m_fage
m_feduc
m_omaps
m_fmaps
m_cigs
m_drink

'''


#######################
# 
lm_full = smf.ols(formula = """bwght ~     birthdata['mage'] +
                                           birthdata['meduc'] +
                                           birthdata['monpre'] +
                                           birthdata['npvis'] +
                                           birthdata['fage'] +
                                           birthdata['feduc'] +
                                           birthdata['omaps'] +
                                           birthdata['fmaps'] +
                                           birthdata['cigs'] +
                                           birthdata['drink'] +
                                           birthdata['male'] +
                                           birthdata['mwhte'] +
                                           birthdata['mblck'] +
                                           birthdata['moth'] +
                                           birthdata['fwhte'] +
                                           birthdata['fblck'] +
                                           birthdata['foth'] +
                                           birthdata['m_meduc'] +
                                           birthdata['m_monpre'] +
                                           birthdata['m_npvis'] +
                                           birthdata['m_fage'] +
                                           birthdata['m_feduc'] +
                                           birthdata['m_omaps'] +
                                           birthdata['m_fmaps'] +
                                           birthdata['m_cigs'] +
                                           birthdata['m_drink'] +
                                           birthdata['out_mage'] +
                                           birthdata['out_meduc'] +
                                           birthdata['out_monpre'] +
                                           birthdata['out_npvis'] +
                                           birthdata['out_fage'] +
                                           birthdata['out_feduc'] +
                                           birthdata['out_omaps'] +
                                           birthdata['out_fmaps'] +
                                           birthdata['out_cigs'] +
                                           birthdata['out_drink'] + -1
                                           """,
                         data = birthdata)


# Fitting Results
results = lm_full.fit()



# Printing Summary Statistics
print(results.summary())



print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

    
    
########################
# Significant Model
########################

lm_significant = smf.ols(formula = """bwght ~    birthdata['mage'] +
                                                 birthdata['monpre']+
                                                 birthdata['fage'] +
                                                 birthdata['omaps'] +
                                                 birthdata['fmaps'] +
                                                 birthdata['male'] +
                                                 birthdata['mwhte'] +
                                                 birthdata['moth'] +
                                                 birthdata['fwhte'] +
                                                 birthdata['fblck'] +
                                                 birthdata['m_fage'] +
                                                 birthdata['m_omaps'] +
                                                 birthdata['m_fmaps'] +
                                                 birthdata['out_omaps'] +
                                                 birthdata['out_fmaps'] +
                                                 birthdata['out_cigs'] + -1
                                           """,
                         data = birthdata)
   
    
# Fitting Results
results = lm_significant.fit()



# Printing Summary Statistics
print(results.summary())



print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

############################################################################
# Machine Learnig
############################################################################

from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling

birth_data = birthdata.drop(['bwght'], axis = 1)


birth_target = birthdata.loc[:, 'bwght']




X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birth_target)

# Let's check to make sure our shapes line up.

# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)




X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birth_target,
            test_size = 0.25,
            random_state = 709)



# Checking shapes again.

# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


########################
# Step 1: Create a model object
########################

# Creating a regressor object
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 1)



# Checking the type of this new object
type(knn_reg)


# Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)



# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)



# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")



# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score)


birthdata.to_excel('birthdata_3.xlsx')   
    

###############################################################################
# How Many Neighbors?
###############################################################################

# This is the exact code we were using before
X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birth_target,
            test_size = 0.25,
            random_state = 709)



# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []



# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


########################
# What is the optimal number of neighbors?
########################

print(test_accuracy)

# Building a model with k = 8
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 8)



# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)



# Scoring the model
y_score = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score)



print(f"""
Our base to compare other models is {y_score.round(3)}.
    
This base helps us evaluate more complicated models and lets us consider
tradeoffs between accuracy and interpretability.
""")


########################
## Does OLS predict better than KNN?
########################

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birth_target,
            test_size = 0.25,
            random_state = 508)



# Prepping the Model
lr = LinearRegression(fit_intercept = False)


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{y_pred.round(2)}
""")



# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))


# Printing model results
print(f"""
Full model KNN score:    {y_score.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")







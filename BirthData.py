#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:25:47 2019

@author: arthurmendes
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import sklearn
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling


# Importing new libraries
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend


# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
from sklearn.model_selection import cross_val_score # k-folds cross validation


file = 'birthweight_feature_set.xlsx'


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



# # Filled with the median

fill = birthdata['meduc'].median()

birthdata['meduc'] = birthdata['meduc'].fillna(fill)


fill = birthdata['npvis'].median()

birthdata['npvis'] = birthdata['npvis'].fillna(fill)


fill = birthdata['feduc'].median()

birthdata['feduc'] = birthdata['feduc'].fillna(fill)


# Checking the overall dataset to see if there are any remaining
# missing values
print(
      birthdata
      .isnull()
      .any()
      .any()
      )


## Scatter plot for flagging
############################

plt.scatter(birthdata['mage'], birthdata['bwght'])

plt.xlabel('Mother Age')

mage_hi = 53

############################

plt.scatter(birthdata['meduc'], birthdata['bwght'])

plt.xlabel('Mother Education')

meduc_lo = 10

############################

plt.scatter(birthdata['monpre'], birthdata['bwght'])

plt.xlabel('Mother Education')

monpre_hi = 5 # =>

############################

plt.scatter(birthdata['npvis'], birthdata['bwght'])

plt.xlabel('Mother Education')

npvis_hi = 20

############################

plt.scatter(birthdata['fage'], birthdata['bwght'])

plt.xlabel('Father Age')

fage_hi = 51

############################

plt.scatter(birthdata['feduc'], birthdata['bwght'])

plt.xlabel('Father Education')

feduc_lo = 10

############################

plt.scatter(birthdata['omaps'], birthdata['bwght'])

plt.xlabel('One minute')

omaps_lo = 5

############################

plt.scatter(birthdata['fmaps'], birthdata['bwght'])

plt.xlabel('Five minute')

fmaps_lo = 7

############################

plt.scatter(birthdata['cigs'], birthdata['bwght'])

plt.xlabel('Cigs per day')

cigs_hi = 20

############################

plt.scatter(birthdata['drink'], birthdata['bwght'])

plt.xlabel('Drinks per week')

drink_hi = 10


########################
# Creating Outlier Flags
########################

# Building loops for outlier imputation



########################
# mage

birthdata['out_mage'] = 0



for val in enumerate(birthdata.loc[ : , 'mage']):

    if val[1] >= mage_hi:
        birthdata.loc[val[0], 'out_mage'] = 1


########################
# meduc

birthdata['out_meduc'] = 0


for val in enumerate(birthdata.loc[ : , 'meduc']):

    if val[1] < meduc_lo:
        birthdata.loc[val[0], 'out_meduc'] = -1


########################
# monpre

birthdata['out_monpre'] = 0


for val in enumerate(birthdata.loc[ : , 'monpre']):

    if val[1] >= monpre_hi:
        birthdata.loc[val[0], 'out_monpre'] = 1


########################
# omaps

birthdata['out_omaps'] = 0


for val in enumerate(birthdata.loc[ : , 'omaps']):

    if val[1] >= omaps_lo:
        birthdata.loc[val[0], 'out_omaps'] = 1


########################
# fmaps

birthdata['out_fmaps'] = 0


for val in enumerate(birthdata.loc[ : , 'fmaps']):

    if val[1] >= fmaps_lo:
        birthdata.loc[val[0], 'out_fmaps'] = 1




########################
# npvis

birthdata['out_npvis'] = 0


for val in enumerate(birthdata.loc[ : , 'npvis']):

    if val[1] > npvis_hi:
        birthdata.loc[val[0], 'out_npvis'] = 1


########################
# fage

birthdata['out_fage'] = 0


for val in enumerate(birthdata.loc[ : , 'fage']):

    if val[1] > fage_hi:
        birthdata.loc[val[0], 'out_fage'] = 1
########################
# feduc

birthdata['out_feduc'] = 0


for val in enumerate(birthdata.loc[ : , 'feduc']):

    if val[1] < feduc_lo:
        birthdata.loc[val[0], 'out_feduc'] = -1


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
lm_price_qual = smf.ols(formula = """bwght ~ birthdata['drink']""",
                         data = birthdata)



# Fitting Results
results = lm_price_qual.fit()


# Printing Summary Statistics
print(results.summary())


# Create a full prediction
birthdata.columns

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
                                           birthdata['m_npvis'] +
                                           birthdata['m_feduc'] +

                                           birthdata['out_mage'] +
                                           birthdata['out_meduc'] +
                                           birthdata['out_monpre'] +
                                           birthdata['out_npvis'] +
                                           birthdata['out_fage'] +
                                           birthdata['out_feduc'] +
                                           birthdata['out_cigs'] +
                                           birthdata['out_drink'] +
                                           birthdata['out_omaps'] +
                                           birthdata['out_fmaps'] +-1
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

# Significant Variables

lm_significant = smf.ols(formula = """bwght ~
                                           birthdata['meduc'] +
                                           birthdata['feduc'] +
                                           birthdata['omaps'] +
                                           birthdata['cigs'] +
                                           birthdata['drink'] +
                                           birthdata['mwhte'] +
                                           birthdata['mblck'] +
                                           birthdata['moth'] +
                                           birthdata['fwhte'] +
                                           birthdata['fblck'] +
                                           birthdata['foth'] +
                                           birthdata['m_meduc'] +
                                           birthdata['m_npvis'] +

                                           birthdata['out_mage'] +
                                           birthdata['out_meduc'] +
                                           birthdata['out_fage'] +
                                           birthdata['out_omaps'] + -1
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

###############################################################################
# Generalization using Train/Test Split
###############################################################################

birth_data   = birthdata.drop(['bwght'],
                                axis = 1)



birthdata_target = birthdata.loc[:, 'bwght']




X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birthdata_target,
            test_size = 0.1,
            random_state = 508)




# Training set
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


###############################################################################
# Forming a Base for Machine Learning with KNN
###############################################################################


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
# The best results occur when k = 4.



# Building a model with k = 5
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 4)



# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score)



print(f"""
Our base to compare other models is {y_score.round(3)}.

This base helps us evaluate more complicated models and lets us consider
tradeoffs between accuracy and interpretability.
""")



# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
birthdata_train = pd.concat([X_train, y_train], axis = 1)



# Review of statsmodels.ols

# Step 1: Build the model
lm_price_qual = smf.ols(formula = """bwght ~ birthdata_train['cigs']""",
                         data = birthdata_train)



# Step 2: Fit the model based on the data
results = lm_price_qual.fit()



# Step 3: Analyze the summary output
print(results.summary())


# Let's pull in the optimal model from before, only this time on the training
# set
lm_significant = smf.ols(formula = """bwght ~
                                           birthdata_train['meduc'] +
                                           birthdata_train['feduc'] +
                                           birthdata_train['omaps'] +
                                           birthdata_train['cigs'] +
                                           birthdata_train['drink'] +
                                           birthdata_train['mwhte'] +
                                           birthdata_train['mblck'] +
                                           birthdata_train['moth'] +
                                           birthdata_train['fwhte'] +
                                           birthdata_train['fblck'] +
                                           birthdata_train['foth'] +
                                           birthdata_train['m_meduc'] +
                                           birthdata_train['m_npvis'] +

                                           birthdata_train['out_mage'] +
                                           birthdata_train['out_meduc'] +
                                           birthdata_train['out_fage'] +
                                           birthdata_train['out_omaps']
                                           """,
                         data = birthdata_train)


# Fitting Results
results = lm_significant.fit()



# Printing Summary Statistics
print(results.summary())


###############################################################################
# Applying the Optimal Model in scikit-learn
###############################################################################

# Preparing a DataFrame based the the analysis above
birth_data   = birthdata.loc[:,['meduc',
                                'feduc',
                                'omaps',
                                'cigs',
                                'drink',
                                'mwhte',
                                'mblck',
                                'moth',
                                'fwhte',
                                'fblck',
                                'foth',
                                'm_meduc',
                                'm_npvis',
                                'out_mage',
                                'out_meduc',
                                'out_fage',
                                'out_omaps']]


# Preparing the target variable
birthdata_target = birthdata.loc[:, 'bwght']


# Same code as before
X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birthdata_target,
            test_size = 0.1,
            random_state = 508)



########################
# Using KNN  on the optimal model (same code as before)
########################

# Exact loop as before
training_accuracy = []
test_accuracy = []



neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)

    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))

    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()



print(test_accuracy)



########################
# The best results occur when k = 17.
########################

# Building a model with k = 17
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 17)



# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score_knn_optimal)



# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)



########################
## Does OLS predict better than KNN?
########################

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birthdata_target,
            test_size = 0.1,
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
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")


###############################################################################
# Outputting Model Coefficients, Predictions, and Other Metrics
###############################################################################

# What does our leading model look like?
pd.DataFrame(list(zip(birth_data.columns, lr.coef_)))


# How well are we predicting on each observation?
pd.DataFrame(list(zip(y_test, lr_pred)))



########################
# Other Metrics
########################

# R-Square (same as the score above)
lr_rsq = sklearn.metrics.r2_score(y_test, lr_pred)
print(lr_rsq)



###############################################################################
# Decision Trees
###############################################################################

# Let's start by building a full tree.
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))


# Creating a tree with 4 levels.
tree_4 = DecisionTreeRegressor(max_depth = 4,
                               random_state = 508)

tree_4_fit = tree_4.fit(X_train, y_train)


print('Training Score', tree_4.score(X_train, y_train).round(3))
print('Testing Score:', tree_4.score(X_test, y_test).round(3))




birthdata.to_excel('birthdata_mani.xlsx')

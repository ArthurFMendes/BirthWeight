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


omaps_lo = 6
omaps_hi = 10


fmaps_lo = 8
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
    
    if val[1] > omaps_hi:
        birthdata.loc[val[0], 'out_omaps'] = 1
########################
# fmaps

birthdata['out_fmaps'] = 0


for val in enumerate(birthdata.loc[ : , 'fmaps']):
    
    if val[1] < fmaps_lo:
        birthdata.loc[val[0], 'out_fmaps'] = -1


for val in enumerate(birthdata.loc[ : , 'fmaps']):
    
    if val[1] > fmaps_hi:
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


########################
# Working with Categorical Variables
########################

# One-Hot Encoding Qualitative Variables
male_dummies = pd.get_dummies(list(birthdata['male']), drop_first = True)

mwhte_dummies = pd.get_dummies(list(birthdata['mwhte']), drop_first = True)

mblck_dummies = pd.get_dummies(list(birthdata['mblck']), drop_first = True)

moth_dummies = pd.get_dummies(list(birthdata['moth']), drop_first = True)

fwhte_dummies = pd.get_dummies(list(birthdata['fwhte']), drop_first = True)

fblck_dummies = pd.get_dummies(list(birthdata['fblck']), drop_first = True)

foth_dummies = pd.get_dummies(list(birthdata['foth']), drop_first = True)



# Concatenating One-Hot Encoded Values with the Larger DataFrame
birthdata_2 = pd.concat(
        [birthdata.loc[:,:],
         male_dummies, mwhte_dummies, mblck_dummies,
         moth_dummies, fwhte_dummies, fblck_dummies, foth_dummies],
         axis = 1)

birthdata_2.columns

lm_full = smf.ols(formula = """bwght ~     birthdata_2['mage'] +
                                           birthdata_2['meduc'] +
                                           birthdata_2['monpre'] +
                                           birthdata_2['npvis'] +
                                           birthdata_2['fage'] +
                                           birthdata_2['feduc'] +
                                           birthdata_2['omaps'] +
                                           birthdata_2['fmaps'] +
                                           birthdata_2['cigs'] +
                                           birthdata_2['drink'] +
                                           birthdata_2['male'] +
                                           birthdata_2['mwhte'] +
                                           birthdata_2['mblck'] +
                                           birthdata_2['moth'] +
                                           birthdata_2['fwhte'] +
                                           birthdata_2['fblck'] +
                                           birthdata_2['foth'] +
                                           birthdata_2['m_meduc'] +
                                           birthdata_2['m_monpre'] +
                                           birthdata_2['m_npvis'] +
                                           birthdata_2['m_fage'] +
                                           birthdata_2['m_feduc'] +
                                           birthdata_2['m_omaps'] +
                                           birthdata_2['m_fmaps'] +
                                           birthdata_2['m_cigs'] +
                                           birthdata_2['m_drink'] +
                                           birthdata_2['out_mage'] +
                                           birthdata_2['out_meduc'] +
                                           birthdata_2['out_monpre'] +
                                           birthdata_2['out_npvis'] +
                                           birthdata_2['out_fage'] +
                                           birthdata_2['out_feduc'] +
                                           birthdata_2['out_omaps'] +
                                           birthdata_2['out_fmaps'] +
                                           birthdata_2['out_cigs'] +
                                           birthdata_2['out_drink'] + -1
                                           """,
                         data = birthdata_2)


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

lm_significant = smf.ols(formula = """bwght ~    birthdata_2['mage'] +
                                                 birthdata_2['monpre']+
                                                 birthdata_2['fage'] +
                                                 birthdata_2['fmaps'] +
                                                 birthdata_2['male'] +
                                                 birthdata_2['mwhte'] +
                                                 birthdata_2['moth'] +
                                                 birthdata_2['fwhte'] +
                                                 birthdata_2['fblck'] +
                                                 birthdata_2['m_fage'] +
                                                 birthdata_2['m_omaps'] +
                                                 birthdata_2['m_fmaps'] +
                                                 birthdata_2['out_omaps'] +
                                                 birthdata_2['out_fmaps'] +
                                                 birthdata_2['out_cigs'] + -1
                                           """,
                         data = birthdata_2)
   
    
# Fitting Results
results = lm_significant.fit()



# Printing Summary Statistics
print(results.summary())



print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    

birthdata.to_excel('birthdata_1.xlsx')   
    
birthdata_2.to_excel('birthdata_Dummies.xlsx')
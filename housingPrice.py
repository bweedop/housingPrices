from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, tree
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from array import array
import seaborn as sns

#Load the training and test data into pandas dataframes.
train = pd.read_csv('train.csv')
preTest = pd.read_csv('test.csv')

#Take a look at what the data consists of...
train.columns()
pd.DataFrame.head(train)
np.shape(train)
#...wow

#Let's take a look at a summary and the distribution of the sale prices.
train['SalePrice'].describe()
train['SalePrice'].skew()
train['SalePrice'].kurt()
sns.distplot(train.SalePrice)
plt.show()
#The sale price is skewed to the the right, 
# mean at about 180k, shows peakedness and positive skew.

#Now, let's look at how SalePrice responds to some of the explanatory variables.
explanatory1 = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[explanatory1]], axis=1)
data.plot.scatter(x=explanatory1, y='SalePrice', ylim=(0,800000))
plt.show()
explanatory2 = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[explanatory2]], axis=1)
data.plot.scatter(x=explanatory2, y='SalePrice', ylim=(0,800000))
plt.show()
#Nice boxplot examples.
explanatory3 = 'OverallQual'
data = pd.concat([train['SalePrice'], train[explanatory3]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=explanatory3, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()
explanatory4 = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[explanatory4]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=explanatory4, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.show()#We can clearly see that SalePrice increases as the overall quality of the
#   house increases and the newer the house...no surprises here ;).

#Now that we have seen some of the explanatory variables and how SalePrice
#   responds to those, how about we check how the others make SalePrice squirm?
#   First, we will make a correlation matrix heatmap...
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
#SalePrice is obviously correlated with variables such as OverallQual, GrLivArea,
#   1stFlrSF, Garage(x), etc. Also, we can see that with these explanatory variables 
#   we have multicollinearity and therefore repetitive information.

#Now, let's just look at the variables that have the highest correlations with 
#   SalePrice.

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables.
# However, as we discussed in the last sub-point, the number of cars that fit into the 
# garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. 
# You'll never be able to distinguish them. Therefore, we just need one of these variables in our 
# analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
# We could say the same for TotalBsmtSF and 1stFloorSF.
#Maybe we should do some time-series analysis to get YearBuilt right? Let's do that now!


#Okay now that we have some features selected, let's look at the correlation matrix
#but this time we will use scatterplots...

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()
#Here we can see some relationships that might have been obvious before but now we have empirical data 
# to backup our intuition that you probably aren't going to buy a house that has a larger basement area
# than GrLivArea...unless you're a prepper.

#Let's deal with our missing data now...Is there a lot missing data? Is there a pattern of data missing?

#sum up all the missing data points from each of the variables...
total = train.isnull().sum().sort_values(ascending=False) 
#Find the percentage of missing data pts for each of the variables...
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#Now, we should not even try to impute values for variables that have 15% or more data pts missing
# and therefore we should probably just act like these variables never even existed...sshhh.

#We can also see that some of the Garage(x) variables are missing a good amount of data as well.
#However, we already decided that GarageCar expresses the most important information.

#Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables are not essential. 
# Furthermore, they have a strong correlation with 'YearBuilt' and 'OverallQual' which are already 
# considered. Thus, we will not lose information if we delete 'MasVnrArea' and 'MasVnrType'.

#Finally, we have one missing observation in 'Electrical'. Since it is just one observation, 
# we'll delete this observation and keep the variable.

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1) #Getting all the variables
# that we found had more than one missing data point and drops them from the train dataset.
train = train.drop(train.loc[train['Electrical'].isnull()].index)#Dropping the one data pt which is missing 
# from the Electrical variable.
train.isnull().sum().max() #We can check that there's no missing data left...

#Scaling our data to look for any outliers in the SalePrice...
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])

#Low and high ranges of our newly scaled data...
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

#Looks like there might be two in the high_range but we will simply keep an eye on these.

#Now we will deal with some of the outliers that maybe you noticed earlier on in the variables such as GrLivArea...
#deleting points.
#Getting the IDs of the data points which should be deleted...
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
#Ahh ID:1299 and 524 are the troublemakers...
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

#Now we will transform our data to meet statistical assumptions...
#We should be meeting four(4) different statistical assumptions in our response and explanatory variables:
#These assumptions are: (1)normality, (2)homoscedasticity, (3)linearity, and (4)absence of multicollinearity and correlated errors.

#Lets start off with SalePrice...
#Raw SalePrice histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm)
pricePlot = stats.probplot(train['SalePrice'], plot=plt)
plt.show()#Okay, not normal but that's fine...this is why we have tranformations...
train['SalePrice'] = np.log(train['SalePrice'])

#transformed SalePrice histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm)
fig = stats.probplot(train['SalePrice'], plot=plt)
scatter(train.SalePrice)
plt.show()
#Raw GrLivArea histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm)
plt.show()
livPlot = stats.probplot(train['GrLivArea'], plot=plt)
plt.show()#Not normal...

#data transformation for GrLivArea
train['GrLivArea'] = np.log(train['GrLivArea'])

#Transformed GrLivArea histogram and normal probability plot...
sns.distplot(train['GrLivArea'], fit=norm)
fig = stats.probplot(train['GrLivArea'], plot=plt)
plt.show()#Normal. Check!

#Raw TotalBsmtSF histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm)
plt.show()
fig = stats.probplot(train['TotalBsmtSF'], plot=plt)
plt.show()#Wow...well let's try to sort this out...

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
train.loc[train['HasBsmt'] == 1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

#histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
plt.show()
stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.show()#Looks sufficient to me...at least better than what it was...

#Now let's check homoscedasticy...
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice'])
plt.show()
#Dummy variables for all of the categorical variables...
train = pd.get_dummies(train)

explanatory1 = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[explanatory1]], axis=1)
data.plot.scatter(x=explanatory1, y='SalePrice', ylim=(0,20))
plt.show()
explanatory2 = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[explanatory2]], axis=1)
data.plot.scatter(x=explanatory2, y='SalePrice', ylim=(0,20))
plt.show()
#Nice boxplot examples.
explanatory3 = 'OverallQual'
data = pd.concat([train['SalePrice'], train[explanatory3]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=explanatory3, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=20)
plt.show()
explanatory4 = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[explanatory4]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=explanatory4, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=20)
plt.xticks(rotation=90)
plt.show()
#Now let's get to the fun stuff...
#Let's set aside the response variable...
target = "SalePrice"
#And the predictor variables...
predictors = [x for x in cols if x not in target]
X_train, X_test, y_train, y_test = train_test_split(train[predictors], train[target], test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

test = preTest[predictors]
test['GrLivArea'] = np.log(test['GrLivArea'])
test['HasBsmt'] = pd.Series(len(test['TotalBsmtSF']), index=test.index)
test['HasBsmt'] = 0 
test.loc[test['TotalBsmtSF']>0,'HasBsmt'] = 1
test.loc[test['HasBsmt']==1,'TotalBsmtSF'] = np.log(test['TotalBsmtSF'])
test = pd.get_dummies(test)
test = test.drop('HasBsmt', axis = 1)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(test)
test = imp.transform(test)
test = pd.DataFrame(data = test, columns=predictors)
predictions = model.predict(test)
predictions = list(np.exp(predictions))
submission = pd.DataFrame({"Id":list(preTest.Id), "SalePrice":predictions})
submission.to_csv('submission.csv', sep = ',', index=False)


model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.2, random_state=1)
model.fit(train[predictors], train[target])
predictions = model.predict(test)
predictions = list(np.exp(predictions))
submission = pd.DataFrame({"Id":list(preTest.Id), "SalePrice":predictions})
submission.to_csv('submission.csv', sep = ',', index=False)

nnModel = MLPRegressor(
    hidden_layer_sizes=(5,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nnFit = nnModel.fit(train[predictors], train[target])
nnPredictions = nnModel.predict(test)
nnPredictions = list(np.exp(nnPredictions))
submission = pd.DataFrame({"Id":list(preTest.Id), "SalePrice":nnPredictions})
submission.to_csv('submission.csv', sep = ',')

import itertools
leafs = list(range(1,21))
feats=[x / 100.0 for x in range(1, 21)]
rs = list(range(1,21))
params = [leafs, feats, rs]
score = []
for i in itertools.product(*params):
    param = i
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=param[0], max_features=param[1], random_state=param[2])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score.append(model.score(X_test, y_test))

results = []
for i in itertools.product(*params):
    results.append(i)

opt = score.index(max(score))
optParams = results[opt]

scale_LCV.fit(train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'])
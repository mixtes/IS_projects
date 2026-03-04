import pandas as pd

import matplotlib .pyplot as plt

import seaborn as sb

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import re

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

pd.set_option('display.width', None)
data = pd.read_csv('../data/titanic_train.csv')

data.insert(loc=0, column='Passenger Id', value=np.arange(0, len(data), 1))

print(data.head())

print(data.info())

print(data.describe())

print(data.describe(include=[object]))

total = data.isnull().sum().sort_values(ascending=False)
perc1 = data.isnull().sum() / data.isnull().count() * 100
perc2 = (round(perc1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, perc2], axis=1, keys=['Total', '%'])
print(missing_data.head(5))

sb.displot(data[data['age'].notna()], x='age', col='sex', hue='survived', hue_order=[0, 1], multiple='fill', bins=16)
plt.show()

grid = sb.FacetGrid(data=data, col='embarked')

grid.map(sb.pointplot, 'pclass', 'survived', 'sex', order=None, hue_order=None)
grid.add_legend()
plt.show()

plt.figure()
sb.barplot(x='pclass', y='survived', data=data)

grid = sb.FacetGrid(data=data, col='pclass', row='survived')
grid.map(plt.hist, 'age', bins=16)
grid.add_legend()
plt.show()

bins = 30
data['qcut_fare'] = pd.qcut(data['fare'], bins)

data = data.sort_values(by='qcut_fare')

data['qcut_fare'] = data['qcut_fare'].astype(str)

sb.displot(data, x='qcut_fare', hue='survived', hue_order=[0, 1], multiple='fill', bins=bins)
plt.xticks(rotation=90)
plt.show()

data.drop(columns=['qcut_fare'], inplace=True)

plt.figure()
data['family'] = data['sibsp'] + data['parch']
data = data.drop(columns=['sibsp', 'parch'])
sb.pointplot(x='family', y='survived', data=data)
plt.show()

cabin = set(map(lambda val: str(val)[0], data['cabin'].unique()))
print(sorted(cabin))

data['deck'] = data['cabin'].apply(lambda val: str(val)[0])
data = data.drop(columns=['cabin'])

sb.displot(data, x='deck', hue='pclass', multiple='fill')
sb.displot(data, x='deck', hue='survived', multiple='fill')
plt.show()

g = ['pclass', 'sex', 'deck', 'family']
data['age'] = data.groupby(by=g)['age'].transform(lambda c: c.fillna(c.mean()))

data['age'] = data['age'].fillna(data['age'].mean())

data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

print(data.info())


data = data.drop(columns=['Passenger Id'])
data = data.drop(columns=['ticket'])

data['age'] = data['age'].astype(int)

data['deck'] = data['deck'].replace(['A', 'B', 'C', 'T'], '1st')
data['deck'] = data['deck'].replace(['F', 'G'], '2nd_3rd')
data['deck'] = data['deck'].replace(['D', 'E'], 'mixed')
data['deck'] = data['deck'].replace(['n'], 'N')

ind_alone = (data['family'] == 0)
ind_small = ((data['family'] >= 1) & (data['family'] <= 3))
ind_med = ((data['family'] >= 4) & (data['family'] <= 6))
ind_big = (data['family'] > 6)

data.loc[ind_alone, 'family'] = 'alone'
data.loc[ind_small, 'family'] = 'small'
data.loc[ind_med, 'family'] = 'medium'
data.loc[ind_big, 'family'] = 'big'

fun_tit = lambda name: re.search(r'(\w+)\.', name).group(1)
data['title'] = data['name'].apply(fun_tit)
titles = set(data['title'].unique())

print(sorted(titles))

female_common = {'Ms', 'Mrs', 'Mme', 'Miss', 'Mlle'}
titles = set.difference(titles, female_common)
titles.remove('Mr')
data.loc[data['title'].isin(titles), 'title'] = 'Rare'
data.loc[data['title'].isin(female_common), 'title'] = 'Mrs'
data = data.drop(columns=['name'])

# usefull if order isn't important
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])

# makes n different columns where only one has value of 1 and others have 0
ohe = OneHotEncoder(dtype=int, sparse_output=False)
embarked = ohe.fit_transform(data['embarked'].to_numpy().reshape(-1, 1))
data.drop(columns=['embarked'], inplace=True)
data = data.join(pd.DataFrame(data=embarked, columns=ohe.get_feature_names_out(['embarked'])))

titles = pd.get_dummies(data['title'], prefix='tit')
data.drop(columns=['title'], inplace=True)
data = data.join(pd.DataFrame(data=titles))

decks = pd.get_dummies(data['deck'], prefix='deck')
data.drop(columns=['deck'], inplace=True)
data = data.join(pd.DataFrame(data=decks))

families = pd.get_dummies(data['family'], prefix='fam')
data.drop(columns=['family'], inplace=True)
data = data.join(pd.DataFrame(data=families))

data.loc[(data['age'] <= 5), 'age'] = 0
data.loc[(data['age'] > 7) & (data['age'] <= 19), 'age'] = 1
data.loc[(data['age'] > 19) & (data['age'] <= 21), 'age'] = 2
data.loc[(data['age'] > 21) & (data['age'] <= 28), 'age'] = 3
data.loc[(data['age'] > 28) & (data['age'] <= 36), 'age'] = 4
data.loc[(data['age'] > 36) & (data['age'] <= 47), 'age'] = 5
data.loc[(data['age'] > 47) & (data['age'] <= 60), 'age'] = 6
data.loc[(data['age'] > 60), 'age'] = 7

data.loc[(data['fare'] > 0) & (data['fare'] <= 7.9), 'fare'] = 0
data.loc[(data['fare'] > 7.9) & (data['fare'] <= 10.5), 'fare'] = 1
data.loc[(data['fare'] > 10.5) & (data['fare'] <= 13), 'fare'] = 2
data.loc[(data['fare'] > 13) & (data['fare'] <= 15.7), 'fare'] = 3
data.loc[(data['fare'] > 15.7) & (data['fare'] <= 26), 'fare'] = 4
data.loc[(data['fare'] > 26) & (data['fare'] <= 27), 'fare'] = 5
data.loc[(data['fare'] > 27) & (data['fare'] <= 52.5), 'fare'] = 6
data.loc[(data['fare'] > 52.5) & (data['fare'] <= 62.5), 'fare'] = 7
data.loc[(data['fare'] > 62.5) & (data['fare'] <= 78), 'fare'] = 8
data.loc[(data['fare'] > 78), 'fare'] = 9

data['fare'] = data['fare'].astype(int)

print(data.head(10))


dtc_model = DecisionTreeClassifier(criterion='entropy')

X = data.drop(columns=['survived'])
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=234, shuffle=True)

dtc_model.fit(X_train, y_train)

labels_predicted = dtc_model.predict(X_test)

ser_pred = pd.Series(data=labels_predicted, name='Predicted', index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_pred], axis=1)

print(res_df.head(10))

print(f'Model score: {dtc_model.score(X_test, y_test):0.3f}')


cv_res = cross_validate(dtc_model, X, y, cv=10)
print(sorted(cv_res['test_score']))
print(f'CV score: {cv_res["test_score"].mean():0.3f}')


fig, axes = plt.subplots(1, 1, figsize=(8, 3), dpi=400)

tree.plot_tree(decision_tree=dtc_model, max_depth=3, feature_names=X.columns, class_names=['Death', 'Life'],
               fontsize=3, filled=True)

fig.savefig('tree.png')


plt.figure()

feat = dict(zip(X.columns, dtc_model.feature_importances_))

items = sorted(feat.items(), key=lambda it: it[1], reverse=True)

keys, values = zip(*items)

plt.bar(x=range(len(keys)), height=values)
plt.xticks(ticks=range(len(keys)), labels=keys, rotation=90)
plt.title('Feature importance')
plt.show()


plt.figure()
sb.heatmap(data.corr(), annot=True, fmt='.2f')
plt.show()

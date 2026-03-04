import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import tree

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)

data = pd.read_csv('../data/cakes_train.csv')

print("First five elements of the dataframe: \n" + str(data.head()) + '\n')
print("Last five elements of the dataframe: \n" + str(data.tail()) + '\n')

print("Dataframe info: ")
data.info()
print('\n')

print("Statistical information about every attribute: ")
print(data.describe())
print('\n')

input_attributes = ['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']
output_attribute = 'type'

plt.figure(figsize=(12, 10))
for i, feature in enumerate(input_attributes):
    plt.subplot(3, 2, i+1)
    sb.boxplot(x=output_attribute, y=feature, data=data)
    plt.xlabel('Cake Type', fontsize=11)
    plt.ylabel(feature, fontsize=11)
    plt.title(f'{feature} by {output_attribute}', fontsize=12)

plt.tight_layout()
plt.show()


plt.figure()
sb.heatmap(data[input_attributes].corr(), annot=True, fmt='.2f')
plt.show()


X = data[input_attributes]
y = data[output_attribute]

dtc_model = DecisionTreeClassifier()
dtc_model.fit(X, y)

importances = pd.Series(dtc_model.feature_importances_, index=input_attributes)
importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importances')
plt.ylabel('Importance')
plt.show()

data_selected = data.drop(columns=['milk', 'baking_powder'])

selected_attributes = ['flour', 'eggs', 'sugar', 'butter']

X = data[selected_attributes]
y = data[output_attribute]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=137, shuffle=True)

dtc_model.fit(X_train, y_train)

types_predicted = dtc_model.predict(X_test)

ser_pred = pd.Series(data=types_predicted, name='Predicted', index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_pred], axis=1)

print(res_df.head(10))

score = dtc_model.score(X_test, y_test)
print(f'Model score: {score:0.3f}')


cv_res = cross_validate(dtc_model, X, y, cv=10)
print(sorted(cv_res['test_score']))
print(f'CV score: {cv_res['test_score'].mean():0.3f}')


error_rate = 1 - score

print(f"Model error rate: {error_rate:.3f}")


fig, axes = plt.subplots(1, 1, figsize=(8, 3), dpi=400)

tree.plot_tree(decision_tree=dtc_model, max_depth=3, feature_names=X.columns, class_names=['Cupcakes', 'Muffins'],
               fontsize=3, filled=True)

fig.savefig('tree.png')

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

iris.feature_names

df  = pd.DataFrame(iris.data, columns= iris.feature_names)

df.head()

df['target'] = iris.target

df.head()

iris.target_names


df[df.target == 1].head()

df['flower_name'] = df.target.apply(lambda x : iris.target_names[x])

df.head()

from matplotlib import pyplot as plt


%matplotlib inline

df0.head()

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color = 'green', marker = '+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color = 'blue', marker = '.')


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color = 'green', marker = '+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color = 'blue', marker = '.')


from sklearn.model_selection import train_test_split

X = df.drop(['target', 'flower_name'], axis = 'columns')


X.head()

y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= 0.2, )

len(X_train)

len(X_test)

from sklearn.svm import SVC
model = SVC()


model.fit(X_train, Y_train)


model.score(X_test, Y_test)


import seaborn as sns


plt.figure(1)
sns.heatmap(df.corr())
plt.title('Correlation ')

corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot = True, ax = ax, cmap = 'coolwarm')



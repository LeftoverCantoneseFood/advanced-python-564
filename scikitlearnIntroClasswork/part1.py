import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model = GaussianNB()

iris = sns.load_dataset('iris')
iris.head()

print(iris.head())

sns.pairplot(iris, hue='species', height=1.5)
print(sns.pairplot(iris, hue='species', height=1.5))

X_iris = iris.drop('species', axis=1)
X_iris.shape
print("rows ", X_iris.shape, " columns")

y_iris = iris['species']
y_iris.shape
print("rows ", y_iris.shape)

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris)

model.fit(xtrain, ytrain)
y_model = model.predict(xtest)

hereWeGo = accuracy_score(ytest, y_model)
print(hereWeGo)

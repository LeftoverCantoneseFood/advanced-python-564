import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Titanic Dataset.csv')

print(df.shape)
print(df.isnull().sum())

df2 = df.dropna(subset = ['age'])
print(df2.isnull().sum())

print(df.head())
df3 = df2.replace(to_replace = "female", value="1") #make women 1 so that women surviving at higher rate = positive correlation
df4 = df3.replace(to_replace = "male", value="0")
print(df4.head())

boatColumn = pd.to_numeric(df4["boat"], errors="coerce")
df5 = df4
df5['boat'] = boatColumn

print(df5.drop(["name", "cabin", "ticket", "embarked", "home.dest"], axis=1).corr().sort_values("survived", ascending=False))

#            pclass  survived       sex       age     sibsp     parch      fare      boat      body
#survived -0.320486  1.000000  0.538000 -0.055512 -0.012213  0.114438  0.249164 -0.035656       NaN
#sex      -0.144695  0.538000  1.000000 -0.063645  0.095267  0.221144  0.187930 -0.021140  0.015730
#fare     -0.565255  0.249164  0.187930  0.178740  0.141184  0.216723  1.000000 -0.466315 -0.043514
#parch     0.017224  0.114438  0.221144 -0.150917  0.374456  1.000000  0.216723  0.001870  0.050902
#sibsp     0.047221 -0.012213  0.095267 -0.243699  1.000000  0.374456  0.141184 -0.089666 -0.100289
#boat      0.674940 -0.035656 -0.021140 -0.294394 -0.089666  0.001870 -0.466315  1.000000       NaN
#age      -0.408106 -0.055512 -0.063645  1.000000 -0.243699 -0.150917  0.178740 -0.294394  0.058809
#pclass    1.000000 -0.320486 -0.144695 -0.408106  0.047221  0.017224 -0.565255  0.674940 -0.034122
#body     -0.034122       NaN  0.015730  0.058809 -0.100289  0.050902 -0.043514       NaN  1.000000

#highest correlation
    #fare - people who pay more to be on the boat are more likely to survive
    #sex - women are more likely to survive than men
    #age - younger passengers are more likely to survive (negative correlation)
    #parch - no idea what parch is
    #pclass - passengers who are in "1"st class, "2"nd class are more likely to survive than the higher-numbered classes (negative correlation)


x = df5[["sex", "fare", "pclass", "parch"]]
y = df5[['survived']]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=67)

didItSurvive = DecisionTreeClassifier(random_state=21)
didItSurvive.fit(xtrain, ytrain)


yPredictions = didItSurvive.predict(xtest)
accuracy = accuracy_score(ytest, yPredictions)
print("testing ", accuracy)

plot_tree(didItSurvive)
plt.show()
#holy shit this thing has like 200 branches

didItSurvive2 = DecisionTreeClassifier(random_state=21, max_depth=3)
didItSurvive2.fit(xtrain, ytrain)

yPredictions2 = didItSurvive2.predict(xtest)
accuracy2 = accuracy_score(ytest, yPredictions2)
print("testing2 ", accuracy2)

plot_tree(didItSurvive2)
plt.show()
#it got worse, but the answer you're fishing for is overfitting; you train so intensely on the training data that you can no longer generalize to new data

didItSurvive3 = RandomForestClassifier(n_estimators = 100, random_state=21)
didItSurvive3.fit(xtrain, ytrain.values.ravel())

yPredictions3 = didItSurvive3.predict(xtest)
accuracy3 = accuracy_score(ytest, yPredictions3)
print("testing3 ", accuracy3)

cm = confusion_matrix(ytest, yPredictions)

print("1st model")
print(cm)

cm2 = confusion_matrix(ytest, yPredictions2)

print("2nd model")
print(cm2)

cm3 = confusion_matrix(ytest, yPredictions3)

print("3rd model")
print(cm3)

#you need to maximize precision. The consequences of abandoning an alive person are much worse than wasting resources looking for a dead person. In this case, you want to maximize recall and get as many POSSIBLE survivors as possbile.

yProbabilities = didItSurvive3.predict_proba(xtest)[:, 1]

auc = roc_auc_score(ytest, yPredictions)
print("AUC Score 1st model:", auc)

auc2 = roc_auc_score(ytest, yPredictions2)
print("AUC Score 2nd model:", auc2)

auc3 = roc_auc_score(ytest, yPredictions3)
print("AUC Score 3rd model:", auc3)

RocCurveDisplay.from_predictions(ytest, yPredictions)
plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.50)")
plt.title("ROC Curve - Random Forest (Best Model, 78% accurate)")
plt.show()

importances = didItSurvive.feature_importances_
feature_names = xtrain.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importances - Random Forest')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

#the three most important features are sex, fare, & passenger class.
#all features are somewhat useful.

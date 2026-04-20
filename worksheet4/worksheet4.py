import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

model = linear_model.LinearRegression(fit_intercept=True)

#1
#Machine learning is the practice of having software "train" on a bunch of data (draw statistical relationships between datapoints), and then use that training to make predictions based on new, unseen data. Traditioally datasets are split between training and testing. We can use any numerical value in this dataset to predict any other numerical value featured in this dataset. The upcoming question asks us to target dancability.

#2
df = pd.read_csv("top2019.csv")

#<bound method DataFrame.info of                        id                                             name              artists  danceability  energy   key  ...  instrumentalness  liveness  valence    tempo  duration_ms  time_signature
#0   6v3KW9xbzN5yKLt9YKDYA                                         Señorita         Shawn Mendes         0.759   0.548   9.0  ...          0.000000    0.0828    0.749  116.967     190800.0             4.0
#1   2Fxmhks0bxGSBdJ92vM42                                          bad guy        Billie Eilish         0.701   0.425   7.0  ...          0.130000    0.1000    0.562  135.128     194088.0             4.0
#2   0RiRZpuVRbi7oqRdSMwhQ    Sunflower - Spider-Man: Into the Spider-Verse          Post Malone         0.755   0.522   2.0  ...          0.000000    0.0685    0.925   89.960     157560.0             4.0
#3   6ocbgoVGwYJhOv1GgI9Ns                                          7 rings        Ariana Grande         0.778   0.317   1.0  ...          0.000000    0.0881    0.327  140.048     178627.0             4.0
#4   2YpeDb67231RjR0MgVLzs                            Old Town Road - Remix            Lil Nas X         0.878   0.619   6.0  ...          0.000000    0.1130    0.639  136.041     157067.0             4.0
#5   0hVXuCcriWRGvwMV1r5Yn                I Don't Care (with Justin Bieber)           Ed Sheeran         0.798   0.675   6.0  ...          0.000000    0.0894    0.842  101.956     219947.0             4.0
#6   7xQAfvXzm3AkraOtGPWIZ                                             Wow.          Post Malone         0.829   0.539  11.0  ...          0.000002    0.1030    0.388   99.960     149547.0             4.0
#7   7qEHsqek33rTcFNT9PFqL                                Someone You Loved        Lewis Capaldi         0.501   0.405   1.0  ...          0.000000    0.1050    0.446  109.891     182161.0             4.0
#8   5w9c2J52mkdntKOmRLeM2                                        Con Calma         Daddy Yankee         0.737   0.860   8.0  ...          0.000002    0.0574    0.656   93.989     193227.0             4.0
#9   2VxeLyX666F8uXCJ0dZF8                                          Shallow            Lady Gaga         0.572   0.385   7.0  ...          0.000000    0.2310    0.323   95.799     215733.0             4.0
#10  2dpaYNEQHiRxtZbfNsse9                                          Happier           Marshmello         0.687   0.792   5.0  ...          0.000000    0.1670    0.671  100.015     214290.0             4.0

#3
print(df.drop(columns=['id']).info())
print(df.drop(columns=['id']).describe())
print(df.drop(columns=['id']).head(10))

#The dumbest possible model: can i categorize danceability by spotify URI...
#print(df[["id", "name", "danceability"]])
#ValueError: could not convert string to float: '6v3KW9xbzN5yKLt9YKDYA' :(
#i guess my dream of linReg danceability based on randomly generated strings is dead
print(df[["key", "tempo", "loudness", "danceability"]])
print(df[["key", "tempo", "loudness", "danceability"]].corr())

#4

#X_music = df.drop(["id", "name", "artists", "danceability", "duration_ms", "time_signature", "instrumentalness", "energy", "liveness", "loudness", "speechiness", "acousticness", "valence", "duration_ms", "time_signature"], axis=1)
#can i make an inclusive list of stuff I want instead of a 10+ exclusive list of things to drop


#old_old_X_music = df[["key", "tempo", "loudness"]] #not enough
#this model sucks. https://media1.tenor.com/m/GCFKCnIqNBEAAAAd/no-correlation.gif

print(df.drop(["id", "name", "artists"], axis=1).corr().sort_values("danceability", ascending=False))

#things that correlate most with danceability
#                 danceability    energy       key  loudness      mode  speechiness  acousticness  instrumentalness  liveness   valence     tempo  duration_ms  time_signature
#danceability          1.000000  0.273068  0.165208  0.178688  0.007662     0.227283     -0.465118          0.153419 -0.088595  0.391774  0.149066    -0.132798             NaN
#valence               0.391774  0.524079 -0.009197  0.452398  0.148433    -0.171795     -0.252026         -0.117525 -0.027293  1.000000 -0.136827     0.029238             NaN
#energy                0.273068  1.000000  0.243397  0.811968  0.042687    -0.094070     -0.432394         -0.233918 -0.053408  0.524079 -0.013653     0.104674             NaN
#speechiness           0.227283 -0.094070  0.011757 -0.336312  0.098718     1.000000      0.021631          0.444664 -0.061797 -0.171795  0.381334     0.011531             NaN
#loudness              0.178688  0.811968  0.019021  1.000000  0.056971    -0.336312     -0.397793         -0.499371 -0.082489  0.452398  0.119046     0.081521             NaN
#key                   0.165208  0.243397  1.000000  0.019021 -0.076022     0.011757     -0.147647          0.152105  0.010617 -0.009197 -0.227342    -0.023579             NaN
#instrumentalness      0.153419 -0.233918  0.152105 -0.499371  0.177337     0.444664      0.213094          1.000000 -0.081667 -0.117525  0.039057    -0.029457             NaN
#tempo                 0.149066 -0.013653 -0.227342  0.119046  0.035271     0.381334     -0.187871          0.039057 -0.214971 -0.136827  1.000000    -0.078091             NaN
#mode                  0.007662  0.042687 -0.076022  0.056971  1.000000     0.098718      0.112079          0.177337  0.113335  0.148433  0.035271     0.162192             NaN
#liveness             -0.088595 -0.053408  0.010617 -0.082489  0.113335    -0.061797      0.121418         -0.081667  1.000000 -0.027293 -0.214971     0.344154             NaN
#duration_ms          -0.132798  0.104674 -0.023579  0.081521  0.162192     0.011531      0.006667         -0.029457  0.344154  0.029238 -0.078091     1.000000             NaN
#acousticness         -0.465118 -0.432394 -0.147647 -0.397793  0.112079     0.021631      1.000000          0.213094  0.121418 -0.252026 -0.187871     0.006667             NaN
#time_signature             NaN       NaN       NaN       NaN       NaN          NaN           NaN               NaN       NaN       NaN       NaN          NaN             NaN

#old_X_music = df[["valence", "energy", "speechiness", "loudness", "key", "instrumentalness"]]
#when I added "instramentalness", a 6th column, it got worse. am i overfitting?
X_music = df[["valence", "energy", "speechiness"]]
#best i can do is 0.0898429319814561 r2 score?? ugh.

X_music.shape
print("rows ", X_music.shape, " columns")

y_music = df['danceability']
y_music.shape
print("rows ", y_music.shape)

xtrain, xtest, ytrain, ytest = train_test_split(X_music, y_music, random_state=42)
#I am so tired right nowwwwwwwwww

Xtrain = scaler.fit_transform(xtrain)
Xtest = scaler.transform(xtest)

model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

firstR2 = r2_score(ytest, y_model)
print("r2 ", firstR2)
print("mse ", mean_squared_error(ytest, y_model))
print("intercept", model.intercept_)

print("coefficient", model.coef_)

print(ytest)
print("-----")
print(y_model)

#r2 score is NEGATIVE 0.8 WTFFFF :(

#5
plt.scatter(ytest, y_model)
plt.axline((0.6, 0.6), slope=1, color='black')
plt.show()
#an optimal model would have most or all points on the line y=x, with little variation. Unforunately the datapoints here are scattered, so this model is probably weak.

#6
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(X_music, y_music, test_size=0.8, random_state=42)

model.fit(xtrain2, ytrain2)
y_model2 = model.predict(xtest2)

secondR2 = r2_score(ytest2, y_model2)
print("r2 ", secondR2)
print("mse ", mean_squared_error(ytest2, y_model2))
#r^2 is -0.10912013867206594, indicating that this model is actually worse than the first one

#7
Xmusic2 = df[["valence", "energy", "speechiness", "loudness", "key", "instrumentalness", "tempo", "mode"]]
xtrain3, xtest3, ytrain3, ytest3 = train_test_split(Xmusic2, y_music, test_size=0.8, random_state=42)

model.fit(xtrain3, ytrain3)
y_model3 = model.predict(xtest3)

thirdR2 = r2_score(ytest3, y_model3)
print("r2 ", thirdR2)
print("mse ", mean_squared_error(ytest3, y_model3))
#DOES THAT SAY -247?? I DIDN'T REALIZE THAT WAS POSSIBLE?


#8
#already did this above to find the optimal x for my model...
print(df.drop(["id", "name", "artists"], axis=1).corr().sort_values("danceability", ascending=False))
sns.heatmap(df.drop(["id", "name", "artists"], axis=1).corr(), annot=True)
plt.show()

#9
#A good model is built with lots of datapoints that are strongly correlated to the target. Proper use of metrics & parameters can help us understand and fine-tine our models.
#For this spotify activitivy specifically, we need more data. It's very difficult to build a good model with only 100 datapoints. I've tried all sorts of combinations and I can't get a good R^2 score. There is also weak correlation between the existing points within this dataset. I need a better dataset.

#10
#Both the MSE & R2s for each model are poor tbh. Performance on W4 is not great. Of all the models, the first model is the only one with a positive R^2 (guessing at random would be more accurate than models 2 & 3), so that is the strongest model.
print("-- final --")
printprint("r2 ", firstR2)
print("mse ", mean_squared_error(ytest, y_model))
print("-----")
print("r2 (2)", secondR2)
print("mse (2)", mean_squared_error(ytest2, y_model2))
print("r2 (3)", thirdR2)
print("mse (3)", mean_squared_error(ytest3, y_model3))

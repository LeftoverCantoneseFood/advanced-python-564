import pandas as pd
import time
import matplotlib.pyplot as plt
df = pd.read_csv("imdb_top_1000.csv")

i = 0;

#1
print("----------------------------")
print("-- Print the 5 first rows --")
for i in range(5):
    print(df.loc[i,:])
    i += 1

#2
print("-----------------------")
print("-- Dataframe Summary --")
print(df.info())

#3 & 4
print("---------------------")
print("-- Dataframe Vomit --")
print(df.describe())

#why tf are python comments done with #hashtags... i hate this. // is better

#               Rank         Year  Runtime (Minutes)       Rating         Votes  Revenue (Millions)   Metascore
#count  1000.000000  1000.000000        1000.000000  1000.000000  1.000000e+03          872.000000  936.000000
#mean    500.500000  2012.783000         113.172000     6.723200  1.698083e+05           82.956376   58.985043
#std     288.819436     3.205962          18.810908     0.945429  1.887626e+05          103.253540   17.194757
#min       1.000000  2006.000000          66.000000     1.900000  6.100000e+01            0.000000   11.000000
#25%     250.750000  2010.000000         100.000000     6.200000  3.630900e+04           13.270000   47.000000
#50%     500.500000  2014.000000         111.000000     6.800000  1.107990e+05           47.985000   59.500000
#75%     750.250000  2016.000000         123.000000     7.400000  2.399098e+05          113.715000   72.000000
#max    1000.000000  2016.000000         191.000000     9.000000  1.791916e+06          936.630000  100.000000

#5
#mean is the (mean) average IMDb rating for all the movies in the dataset. On average, a movie is rated 6.72/10.
#minimum is the lowest IMDb rating in the dataset, which is 1.9. This movie is lower rated than all others. Yikes.
#75% is the rating of third quartile, Q3. This tells us that 75% of movies have a rating <=7.4, and 25% of movies are rated >7.4. A movie rated above 7.4% would be in the top 25% of this dataset.

#6
df["Runtime (Minutes)"].plot.hist(bins=15)
plt.xlabel("Movie Runtime")
plt.title("Histogram of IMDb Movie Runtimes")
#plt.show(block=False) #run the rest of the script in the background
plt.show() #nvm, that sidn't work.
#The majority of runtimes appear to be clustered between 90 & 130 minutes. The average is 113, median is mode is 110, the median is 128.5 minutes.

#7
#print("Standard Deviation " + df["Rating"].std()) #this tells us how far a rating is from the mean. Standard deviation is an indication of how "spread out" the dataset is.
#print("Variance " + df["Rating"].var()) #the variance is similar to standard deviation, but it is given in squared units, while standard deviation is in the same units as data.
#using commas to add strings together is dumb, but FINE. I GUESS BRO.\
print("-----------------------------------")
print("-- Standard Deviation & Variance --")
print("Standard Deviation ", df["Rating"].std())
print("Variance ", df["Rating"].var())

#8
#the standard deviation tells us how far a rating is from the mean. Standard deviation is an indication of how "spread out" the dataset is.
#the variance is similar to standard deviation, but it is given in squared units, while standard deviation is in the same units as data.

#9
print("-----------------------------")
print("-- 10 Highest Rated Movies --")
top_10_array = df.sort_values(by="Rating", ascending=False).head(10)
print(top_10_array) #damn, Christopher Nolan with 4/10. Good for him. Dark Knight was filmed outside my childhood house :)

#10
print("---------------------------")
print("-- 5 Lowest Rated Movies --")
bottom_5_array = df.sort_values(by="Rating").head(5)
print(bottom_5_array)

#11
print("-----------------------------")
print("-- Movies above 8.5 rating --")
print(df[df["Rating"] > 8.5])

#12
print("------------------------")
print("-- Correlation Scores --")
print(df[["Votes", "Runtime (Minutes)", "Rating"]])
print(df[["Votes", "Runtime (Minutes)", "Rating"]].corr())
#all columns are positively correlated

#13
print("-------------------------------------------------------------")
print("-- Movies shorter than 100 minutes & rated higher than 8.0 --")
print(df[(df["Rating"] > 8.0) & (df["Runtime (Minutes)"] < 100)])
#there is a possitive correlation between movie runtimes and ratings. This would suggest that shorter runtime movies have worse ratings?

#14
plt.close('all')
plt.boxplot(df["Rating"])
plt.ylabel("Rating")
plt.title("Boxplot of IMDB Movie Ratings")
plt.show()
#yes, there is a collection outliers below the 25% mark. There are no highly rated outliers, but there is one movie rated really poorly at less than 2/10.

#15
print("----------------------")
print("-- Ratings by Genre --")
genreAverage = df.groupby("Genre")["Rating"].mean()
print(genreAverage.sort_values(ascending=False).head(1))
#Animation, Drama, & Fantasy have the highest ratings, tied at 8.6

#16
print("-----------------------------")
print("-- Incomplete Data Cleanup --")
df.isnull().sum()
print("Complete rows leftover:", df.dropna().shape[0])
#868 rows have complete data.

#17
print("---------------------")
print("-- Ratings by Year --")
yearAverage = df.groupby("Year")["Rating"].mean()
print(yearAverage.sort_values(ascending=False).head(10))

#18
plt.close('all')
plt.plot(yearAverage)
plt.xlabel("Year Released")
plt.ylabel("Rating")
plt.title("Linechart of Movie Ratings / Year")
plt.show()

#19
#I checked the .CSV manually. There is no certificates column?
#print("--------------------------------")
#print("-- 3 Most Common MPA Ratings  --")print(df.value_counts("Certificate").sort_values().head(3))

#20
print("------------------------------")
print("-- Runtime Outlier Analysis --")
runtimeMean = df["Runtime (Minutes)"].mean()
runtimeStandardDeviation = df["Runtime (Minutes)"].std()
runtimeOutliers = df[(df["Runtime (Minutes)"] > runtimeMean + runtimeStandardDeviation)]
print("Mean Runtime ", runtimeMean)
print("Runtime Standard Deviation ", runtimeStandardDeviation)
print("There are ", runtimeOutliers.shape[0], "Movies two standard deviations above the mean.")
print("A Sampling Of Unusually Long Movies")
print(runtimeOutliers.drop(columns=["Actors", "Description", "Genre", "Metascore"]).sort_values("Runtime (Minutes)", ascending=False).head(10)) #drop all random crap so that we can actually see the runtime column when the dataframe is printed.

#21
print("---------------------------------")
print("-- Most Popular Movie By Genre --")
#there are too many genres to print all of them, and I also disagree with the idea of combining like 5+ genres to describe a single movie.
singleGenresOnly = df[~df["Genre"].str.contains(",")] #~ for NO in pandas boolean
popularMovieByGenre = singleGenresOnly.groupby("Genre")["Votes"].idxmax() #idxmax instead of just max() so that we can identify the row by its vote count
print(singleGenresOnly.loc[popularMovieByGenre, ["Genre", "Title", "Votes"]].sort_values("Votes", ascending=False))

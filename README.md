# DATA-ANALYTICS-PROJECT-Indian-Restaurants
Exploratory Data Analysis (EDA) serves as an initial phase in the realm of Machine Learning, extensively employed to gain a profound understanding of the dataset
## About This Project :
In this project, we aim to analyze Zomato restaurant data to identify key factors
 that contribute to the success of restaurants, as measured by their ratings. By
 exploring various features such as location, cuisine, pricing, and service
 offerings, we aim to provide insights that can help restaurant owners and
 Zomato users make informed decisions

## Project Flow: -
1. Data collection and Data loading
2. Data Preprocessing - Handling missing values, Handling outlier, duplicates, Handling categorical(lastly)
3. EDA - Exploratory Data Analysis - Formulate 10-15 questions - based on given problem statement
4. Observation - answer to these 10-15 questions
5. Recommendations - sumaarization based on acquired answers
6. Conclusion - 4-5 point 

## 1. Data collection and Data loading

## importing libraries
import os ## optional library - used to import paths for different files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## warning libarary
import warnings
warnings.filterwarnings("ignore")

## load the dataset
df = pd.read_csv("Indian-Resturants.csv")

## showing the data
df.head(2)

###  Data Overview:
 Explore the basic characteristics of the dataset, including dimensions, data
 types, and missing values

df.shape

df.info()

## 2. Data Preprocessing

## missing values
df.isnull().sum()

## Handling missing values - we will check the percentage of missing values:-
- if missing values are greater than 25% we will drop the column (default you are domain expert)
- if less than 25% values are missing then we will cap the values using statiscal method

## percentage of missing values in each column
(df.isnull().sum()/len(df))*100

## first we will drop zipcode column as it is missing more than 76% of values
df.drop("zipcode", axis = 1, inplace = True)

## Required capping column names = address,cuisines, timings, opentable_support

df.dtypes

## Handling missing in categorical variable
list_of_cols_cat = ["address","cuisines", "timings"]
for i in list_of_cols_cat:
    df[i] = df[i].fillna(df[i].mode()[0])

df["opentable_support" ].value_counts()

## opentable_support column has only 0 as number and it will not be helpful for analysis i will drop the column
df.drop("opentable_support", axis = 1, inplace = True)

df.isnull().sum()

df.info()

num_col1 = ["average_cost_for_two", "price_range", "votes", "photo_count"]
for i in num_col1:
    sns.boxplot(df[i])
    plt.show()

## Handling Outliers

## IQR
num_col1 = ["average_cost_for_two", "price_range", "votes", "photo_count"]
for i in num_col1:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3 - q1
    print(f"Q1:{q1}")
    print(f"Q3:{q3}")
    print(f"IQR:{iqr}")

## formulate UL and lower LL
    UL = q3+1.5*iqr
    LL = q1-1.5*iqr
    df[i] = np.where(df[i]>UL,UL,
                    np.where(df[i]<LL,LL,
                            df[i]))

num_col1 = ["average_cost_for_two", "price_range", "votes", "photo_count"]
for i in num_col1:
    sns.boxplot(df[i])
    plt.show()

data_preprocessd = df.copy()
df.to_csv("data_preprocessd.csv")

## Basic Statistics:
 Calculate and visualize the average rating of restaurants.
 Analyze the distribution of restaurant ratings to understand the overall rating
 landscape.
 1. Statistical Analysis
 2. Univariate - analysis using single column/ feature in dataset
 3. Bivariate analysis -  analysis using two features/columns in dataset
 4. Mulitivariate analysis -  analysis using more than two features/columns in dataset

## summarize dataset
df.describe()

## Average rating
print("Average Rating given by customers:", df["aggregate_rating"].mean())
## plot the ratings
sns.histplot(df["aggregate_rating"], bins = 30, kde = True)
plt.title("Distribution of ratings")
plt.show()

## Location analysis
 1.  Identify the city with the highest concentration of restaurants.
 2. Visualize the distribution of restaurant ratings across different cities

city_count= df["city"].value_counts().head(10)
city_count.plot(kind = "barh", color = "skyblue")
plt.title("Top 10 Cities with highest concentration of restaurants")
plt.show()

## Rating vs city
sns.barplot(x = "city", y = "aggregate_rating", data = df[df["city"].isin(city_count.index)])
plt.xticks(rotation = 45)
plt.title("Top 10 popular cities according ratings")
plt.show()

## Price Range and Rating:
1. Analyze the relationship between price range and restaurant ratings.
2. Visualize the average cost for two people in different price categories

## Q1.
sns.boxplot(df, x = "price_range", y = "aggregate_rating")
plt.title("Price range vs Rating")
plt.xticks(rotation = 45)
plt.show()

## pie chart - when you have distribution based categories - 2 category
## scatter plot - both should numerical
## bar plot - one should be categorical and another should be numerical
## histplot - probability of distribution
# line chart - analysis with time, profit/loss, price- time

## Word Cloud for Reviews:
 1. Create a word cloud based on customer reviews to identify common positive
 and negative sentiments.
 2. Analyze frequently mentioned words and sentiments

df.head(2)

# !pip install wordcloud

from wordcloud import WordCloud

if "rating_text" in df.columns:
    review_txt = ' '.join(df["rating_text"].dropna().tolist())
    wordcloud = WordCloud(width = 800, height = 400, background_color = "white").generate(review_txt)
    plt.figure(figsize = (12,8))
    plt.imshow(wordcloud, interpolation =  "bilinear")
    plt.axis("off")
    plt.title("Word cloud of Customer Reviews")
    plt.show()





## Restaurant Features:
1. Analyze the distribution of restaurants based on features like Wi-Fi, Alcohol
 availability, etc.
2. Investigate if the presence of certain features correlates with higher ratings

df["highlights"].head()

## Abstract syntax library - It segregates different literals for a common data type. Functions as eval but it has more
# strong bond with data type
import ast

df["highlights"] = df["highlights"].apply(ast.literal_eval)

## seaparte unique features for each list
all_feat = set([j for i in df["highlights"] for j in i])

## One hot encoding 
for i in all_feat:
    df[i] = df["highlights"].apply(lambda x: 1 if i in x else 0)



df.columns

df["Wifi"].value_counts()

## if restaurats have wifi
def plot(df, x):
    data = df.groupby(x)["aggregate_rating"].mean()
    sns.barplot(data)
    plt.title(f"{x} vs aggregate_rating")
    plt.show()

plot(df=df, x="Wifi")

## No Alcohol Available	
no_alco_rating = df.groupby("No Alcohol Available")["aggregate_rating"].mean()
sns.barplot(no_alco_rating)

data_preprocessd.info()



plot(df=df, x="Table booking for Groups")

num_col = data_preprocessd.select_dtypes(include = ["int","float64"])
num_col.head(2)

## Correlation - its multivariate analysis
plt.figure(figsize=(14,12))
sns.heatmap(num_col.corr(), annot = True, cmap = "viridis")
plt.title("Correlation between features")
plt.show()

new_data = pd.read_csv("data_preprocessd.csv")
new_data.head()

new_data.info()

 ## Investigate if there's a correlation between the variety of cuisines offered and restaurant ratings.

new_data["Cusine_count"] = new_data["cuisines"].apply(lambda x: len(str(x).split(","))) ## North Indian, Chinese, Continental, Healthy Food
sns.scatterplot(x = new_data["Cusine_count"], y = "aggregate_rating", data =new_data )
plt.show()


new_data["cuisines"].value_counts()

new_data["Cusine_count"]

new_data.head(2)




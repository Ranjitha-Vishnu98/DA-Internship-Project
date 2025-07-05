# import and Load dataset:

import pandas as pd
us_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/DA Internships/CSV files sample/USvideos.csv')
print(us_df.shape)           
print(us_df.columns)           
print(us_df.head(5))           
print(us_df.info())            
print(us_df.isnull().sum())    

# Data Cleaning process:

us_df.columns = us_df.columns.str.lower().str.replace(" ", "_")
us_df.drop_duplicates(inplace=True)
us_df.dropna(subset=['title', 'views'], inplace=True)
us_df['trending_date'] = pd.to_datetime(us_df['trending_date'], format='%y.%d.%m')
us_df['publish_time'] = pd.to_datetime(us_df['publish_time'])
us_df = us_df.drop_duplicates()
us_df['publish_date'] = us_df['publish_time'].dt.date
us_df['publish_hour'] = us_df['publish_time'].dt.hour

# Mapping concepts:

import json

with open(r'C:\Users\admin\OneDrive\Desktop\DA Internships\CSV files sample\US_category_id.json') as f:
    categories = json.load(f)

    category_mapping = {int(item['id']): item['snippet']['title'] for item in categories['items']}
us_df['category'] = us_df['category_id'].map(category_mapping)

# Sentiment Analysis on Video Titles:

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

us_df['title_sentiment'] = us_df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify as Positive/Negative/Neutral

def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    return 'Neutral'

us_df['sentiment_label'] = us_df['title_sentiment'].apply(get_sentiment)

# Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

top_cats = us_df.groupby('category')['views'].mean().sort_values(ascending=False).head(50)
top_cats.plot(kind='barh', title='Top Categories by Avg Views')
plt.xlabel('Average Views')
plt.show()

sns.countplot(x='sentiment_label', data=us_df)
plt.title('Sentiment Distribution of Titles')
plt.show()

sns.heatmap(us_df[['views', 'likes', 'comment_count']].corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

us_df.to_csv('USvideos_cleaned.csv', index=False)
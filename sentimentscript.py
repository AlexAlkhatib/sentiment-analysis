"""
Created on Sun Apr 14 10:08:12 2024

@author: Alex Alkhatib
"""

# =============================================================================
# Importing Libraries
# =============================================================================
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from langdetect import detect
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from textblob import TextBlob
import seaborn as sns

# Reading the CSV File
df = pd.read_csv('chatgpt1.csv')

# Creating a function to detect languages
text = df['Text'][1]
lang = detect(text)

def det(text):
    try:
        lang = detect(text)
    except:
        lang = 'Other'
    return lang

df['Lang'] = df['Text'].apply(det)


# =============================================================================
# Cleaning Text
# =============================================================================

df = df.loc[df['Lang'] == 'en']
df = df.reset_index(drop=True)

# Cleaning some data
df['Text'] = df['Text'].str.replace('https', '')
df['Text'] = df['Text'].str.replace('http', '')
df['Text'] = df['Text'].str.replace('t.co', '')

# Developing a sentiment function

def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Add the function to our dataframe

df['Sentiment'] = df['Text'].apply(get_sentiment)

# Generating a word cloud

comment_words = ''
stopwords = set(STOPWORDS)

for val in df.Text:
    val = str(val)
    tokens = val.split()
    comment_words = comment_words + " ".join(tokens)+ " "

wordcloud = WordCloud(width=900, height=500, background_color = 'black', 
                      stopwords = stopwords, min_font_size=10).generate(comment_words)

plt.figure(figsize = (8,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout()
plt.show()


# Creating a bar chart

sns.set_style('whitegrid')
plt.figure(figsize=(10,5))

sns.countplot(x = 'Sentiment' , data = df)
plt.xlabel('Sentiment')
plt.ylabel('Count of Sentiment')
plt.title('Sentiment Distribution')
plt.show()


    
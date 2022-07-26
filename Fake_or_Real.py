import pandas as pd
import numpy as np
import nltk
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import os
from textblob import TextBlob

#cwd = os.getcwd()
#os.chdir("C:/Users/Carlo/Desktop/UNI/Applied Machine Learning/datasets")

df = pd.read_csv("WELFAKE_Dataset.csv", sep=",", low_memory=False)


wc = WordCloud().generate(df.groupby('label')['title'].sum()[1])
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

print("Info\n")
df.info()
print("Head\n")
print(df.head())
print("Shape\n")
print(df.shape)
print("Dtype\n")
print(df.dtypes)
print("Tail\n")
print(df.tail())
print("Describe\n")
print(df.describe())
print("Columns\n")
print(df.columns)

#Column 0 = Index (int)
#Column 1 = title (string)
#Column 2 = text (string)
#Column 3 = labeled news as fake or real (fake = 1, real = 0)

print(df.label.value_counts())

#There are 37106 fake articles and 35028 real articles in the dataset
print("Title null entries")
print(df.title.isnull().sum())

#There are 558 entries without a title
print("Text null entries")
print(df.text.isnull().sum())

#And 39 entries without text

df1 = df[df.isna().any(axis=1)] #saves all entries with no title and/or no text in df1

#adds title_length and text_length to dataframe
title_length = []
text_length = []
for i in range(0,len(df.index)):
    title_length.append(len(str(df.loc[i, "title"])))
    text_length.append(len(str(df.loc[i, "text"])))
df["title_length"] = title_length
df["text_length"] = text_length

#for i in range(len(df.index)):
#    if df.loc[i, "title_length"] >= 50:
#        print(df.loc[i, "title"])

#One big problem: Due to the data being obtained through an automated process there are sometimes missing whitespaces and/or links, \n or other non-text inputs
mean_word_length_text = []
num_of_words_text = []
mean_word_length_title = []
num_of_words_title = []
for i in range(0,len(df.index)):
    text_row = str(df.loc[i, "text"]).split()
    num_of_words_text.append(len(text_row))
    try:
        mean_word_length_text.append(sum(map(len, text_row))/len(text_row))
    except ZeroDivisionError:
        mean_word_length_text.append(0)
    title_row = str(df.loc[i, "title"]).split()
    num_of_words_title.append(len(title_row))
    try:
        mean_word_length_title.append(sum(map(len, title_row))/len(title_row))
    except ZeroDivisionError:
        mean_word_length_title.append(0)
        continue



df["title_wordnum"] = num_of_words_title
df["text_wordnum"] = num_of_words_text
df["title_meanlen"] = mean_word_length_title
df["text_meanlen"] = mean_word_length_text

for i in range(len(df.index)):
    if df.loc[i, "title_wordnum"] >= 60:
        print(df.loc[i, "title"])

#Plot of text/title length and text/title mean word length in one subplot
fig, axes = plt.subplots(2,2, figsize=(12,12))
sb.histplot(ax=axes[1, 0], data=df, x="text_length", kde=True)
axes[1, 0].set_xlim(0, 30000)
axes[1, 0].set_xlabel("Length of Text [characters]")
sb.histplot(ax=axes[0, 0], data= df, x="title_length", kde=True)
axes[0, 0].set_xlim(0,)
axes[0, 0].set_xlabel("Length of Title [characters]")
sb.histplot(ax=axes[1, 1], data=df, x="text_meanlen", kde=True)
axes[1, 1].set_xlim(0,25)
axes[1, 1].set_xlabel("Mean word length [text]")
sb.histplot(ax=axes[0, 1], data=df, x="title_meanlen", kde=True)
axes[0, 1].set_xlim(0,25)
axes[0, 1].set_xlabel("Mean word length [title]")

#plt.show()

df_fake = df.loc[df["label"] == 1]
df_real = df.loc[df["label"] == 0]
#Plotting of differences in title length and mean word length by Fake/Real News categorization
fig, axes = plt.subplots(2,2, figsize=(12,12))
sb.histplot(ax=axes[0, 0], data= df_real, x="title_wordnum", kde=True)
axes[0, 0].set_xlim(0,)
axes[0, 0].set_ylim(0,)
axes[0, 0].set_xlabel("Length of Title [words]")
axes[0, 0].set_title("Real News")
sb.histplot(ax=axes[0, 1], data=df_fake, x="title_wordnum", kde=True)
axes[0, 1].set_xlim(0,)
axes[0, 1].set_ylim(0,)
axes[0, 1].set_xlabel("Length of Title [words]")
axes[0, 1].set_title("Fake News")
sb.histplot(ax=axes[1, 0], data=df_real, x="title_meanlen", kde=True)
axes[1, 0].set_xlim(0,15)
axes[1, 0].set_ylim(0,1500)
axes[1, 0].set_xlabel("Mean word length [title]")
sb.histplot(ax=axes[1, 1], data=df_fake, x="title_meanlen", kde=True)
axes[1, 1].set_xlim(0,15)
axes[1, 1].set_ylim(0,1500)
axes[1, 1].set_xlabel("Mean word length [title]")


#plt.show()

#Plotting of differences in text length and mean word length by Fake/Real News categorization

fig, axes = plt.subplots(2,2, figsize=(12,12))
sb.histplot(ax=axes[0, 0], data= df_real, x="title_wordnum", kde=True)
axes[0, 0].set_xlim(0,)
axes[0, 0].set_ylim(0,)
axes[0, 0].set_xlabel("Length of text [words]")
axes[0, 0].set_title("Real News")
sb.histplot(ax=axes[0, 1], data=df_fake, x="title_wordnum", kde=True)
axes[0, 1].set_xlim(0,)
axes[0, 1].set_ylim(0,)
axes[0, 1].set_xlabel("Length of text [words]")
axes[0, 1].set_title("Fake News")
sb.histplot(ax=axes[1, 0], data=df_real, x="text_meanlen", kde=True)
axes[1, 0].set_xlim(0,15)
axes[1, 0].set_ylim(0,1300)
axes[1, 0].set_xlabel("Mean word length [text]")
sb.histplot(ax=axes[1, 1], data=df_fake, x="text_meanlen", kde=True)
axes[1, 1].set_xlim(0,15)
axes[1, 1].set_ylim(0,1300)
axes[1, 1].set_xlabel("Mean word length [text]")

#plt.show()
pd.set_option('display.max_columns', None)
#print(title_length, num_of_words_title, mean_word_length_title)
print("New:\n", df.describe(include="all"))

# Removing HTML

def remove_html(text)
    soup = BeautifulSoup(text, "lxml")
    html_free = soup.get_text()
    return html_free

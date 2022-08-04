#wtf git
import pandas as pd

import seaborn as sb
import matplotlib.pyplot as plt

from langdetect import detect
import re
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
#from readability import Readability
from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud
import os

from langdetect import DetectorFactory
from bs4 import BeautifulSoup
import string

# from textblob import TextBlob
# from matplotlib.ticker import FormatStrFormatter
# import numpy as np


#cwd = os.getcwd()
#os.chdir(r"C:\Users\Jens\Desktop\Unizeug\Master\2. Semester\Applied Machine Learning\Final project")
df = pd.read_csv("WELFAKE_Dataset_modified.csv", sep=",", low_memory=False, nrows=1000)
# df = pd.read_csv("WELFAKE_Dataset_modified.csv", sep=",", low_memory=False)

# DetectorFactory.seed = 0
# language_list = []
# for i in range(0, len(df.index)):
#     try:
#         language = detect(df.loc[i, "text"])
#         if language == "":
#             language = "error"
#     except:
#         language = "unknown"
#     language_list.append(language)
#
# print(len(language_list))
# df["language"] = language_list
# df.to_csv("WELFAKE_Dataset_modified.csv", index=False)

# print(df["language"].value_counts())
# for i in range(0,len(df.index)):
#    if df.loc[i, "language"] == "unknown":
#        print(df.iloc[i])


# wc = WordCloud().generate(df.groupby('label')['title'].sum()[1])
# plt.figure(figsize=(15, 15))
# plt.imshow(wc, interpolation='bilinear')
# plt.axis("off")
# # plt.show()
#
# print("Info\n")
# df.info()
# print("Head\n")
# print(df.head())
# print("Shape\n")
# print(df.shape)
# print("Dtype\n")
# print(df.dtypes)
# print("Tail\n")
# print(df.tail())
# print("Describe\n")
# print(df.describe())
# print("Columns\n")
# print(df.columns)

# Column 0 = Index (int)
# Column 1 = title (string)
# Column 2 = text (string)
# Column 3 = labeled news as fake or real (fake = 1, real = 0)

print(df.label.value_counts())

# There are 37106 fake articles and 35028 real articles in the dataset
print("Title null entries")
print(df.title.isnull().sum())

# There are 558 entries without a title
print("Text null entries")
print(df.text.isnull().sum())

# And 39 entries without text

df1 = df[df.isna().any(axis=1)]  # saves all entries with no title and/or no text in df1

# adds title_length and text_length to dataframe
title_length = []
text_length = []
for i in range(0, len(df.index)):
    title_length.append(len(str(df.loc[i, "title"])))
    text_length.append(len(str(df.loc[i, "text"])))
df["title_length"] = title_length
df["text_length"] = text_length

# for i in range(len(df.index)):
#    if df.loc[i, "title_length"] >= 50:
#        print(df.loc[i, "title"])

# One big problem: Due to the data being obtained through an automated process there are sometimes missing whitespaces and/or links, \n or other non-text inputs
mean_word_length_text = []
num_of_words_text = []
mean_word_length_title = []
num_of_words_title = []
for i in range(0, len(df.index)):
    text_row = str(df.loc[i, "text"]).split()
    num_of_words_text.append(len(text_row))
    try:
        mean_word_length_text.append(sum(map(len, text_row)) / len(text_row))
    except ZeroDivisionError:
        mean_word_length_text.append(0)
    title_row = str(df.loc[i, "title"]).split()
    num_of_words_title.append(len(title_row))
    try:
        mean_word_length_title.append(sum(map(len, title_row)) / len(title_row))
    except ZeroDivisionError:
        mean_word_length_title.append(0)
        continue

df["title_wordnum"] = num_of_words_title
df["text_wordnum"] = num_of_words_text
df["title_meanlen"] = mean_word_length_title
df["text_meanlen"] = mean_word_length_text





# Sentiment Analysis
def sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_dict = sia.polarity_scores(text)
    return sentiment_dict['compound']

df["sentiment score"] = df["text"].apply(lambda x: sentiment_score(x))


for i in range(len(df.index)):
    if df.loc[i, "title_wordnum"] >= 60:
        print(df.loc[i, "title"])

# # Plot of text/title length and text/title mean word length in one subplot
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# sb.histplot(ax=axes[1, 0], data=df, x="text_length", kde=True)
# axes[1, 0].set_xlim(0, 30000)
# axes[1, 0].set_xlabel("Length of Text [characters]")
# sb.histplot(ax=axes[0, 0], data=df, x="title_length", kde=True)
# axes[0, 0].set_xlim(0, )
# axes[0, 0].set_xlabel("Length of Title [characters]")
# sb.histplot(ax=axes[1, 1], data=df, x="text_meanlen", kde=True)
# axes[1, 1].set_xlim(0, 25)
# axes[1, 1].set_xlabel("Mean word length [text]")
# sb.histplot(ax=axes[0, 1], data=df, x="title_meanlen", kde=True)
# axes[0, 1].set_xlim(0, 25)
# axes[0, 1].set_xlabel("Mean word length [title]")

# plt.show()

# df_fake = df.loc[df["label"] == 1]
# df_real = df.loc[df["label"] == 0]
# # Plotting of differences in title length and mean word length by Fake/Real News categorization
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# sb.histplot(ax=axes[0, 0], data=df_real, x="title_wordnum", kde=True)
# axes[0, 0].set_xlim(0, )
# axes[0, 0].set_ylim(0, )
# axes[0, 0].set_xlabel("Length of Title [words]")
# axes[0, 0].set_title("Real News")
# sb.histplot(ax=axes[0, 1], data=df_fake, x="title_wordnum", kde=True)
# axes[0, 1].set_xlim(0, )
# axes[0, 1].set_ylim(0, )
# axes[0, 1].set_xlabel("Length of Title [words]")
# axes[0, 1].set_title("Fake News")
# sb.histplot(ax=axes[1, 0], data=df_real, x="title_meanlen", kde=True)
# axes[1, 0].set_xlim(0, 15)
# axes[1, 0].set_ylim(0, 1500)
# axes[1, 0].set_xlabel("Mean word length [title]")
# sb.histplot(ax=axes[1, 1], data=df_fake, x="title_meanlen", kde=True)
# axes[1, 1].set_xlim(0, 15)
# axes[1, 1].set_ylim(0, 1500)
# axes[1, 1].set_xlabel("Mean word length [title]")

# plt.show()

# Plotting of differences in text length and mean word length by Fake/Real News categorization

# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# sb.histplot(ax=axes[0, 0], data=df_real, x="title_wordnum", kde=True)
# axes[0, 0].set_xlim(0, )
# axes[0, 0].set_ylim(0, )
# axes[0, 0].set_xlabel("Length of text [words]")
# axes[0, 0].set_title("Real News")
# sb.histplot(ax=axes[0, 1], data=df_fake, x="title_wordnum", kde=True)
# axes[0, 1].set_xlim(0, )
# axes[0, 1].set_ylim(0, )
# axes[0, 1].set_xlabel("Length of text [words]")
# axes[0, 1].set_title("Fake News")
# sb.histplot(ax=axes[1, 0], data=df_real, x="text_meanlen", kde=True)
# axes[1, 0].set_xlim(0, 15)
# axes[1, 0].set_ylim(0, 1300)
# axes[1, 0].set_xlabel("Mean word length [text]")
# sb.histplot(ax=axes[1, 1], data=df_fake, x="text_meanlen", kde=True)
# axes[1, 1].set_xlim(0, 15)
# axes[1, 1].set_ylim(0, 1300)
# axes[1, 1].set_xlabel("Mean word length [text]")

# plt.show()

# Let's check how well the data is balanced between fake and real news and plot the respective sentiment scores
# fake_count = len(df_fake) / df.shape[0]
# real_count = len(df_real) / df.shape[0]

# label_count = [fake_count, real_count]
#
# df["pos_or_neg"] = df["sentiment score"].apply(
#     lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))
#
# plt.pie(x=label_count, explode=[0.1, 0.1], colors=['firebrick', 'navy'], startangle=90, shadow=True,
#             labels=['Fake News', 'True News'], autopct='%1.1f%%')
# plt.show()

# ax = sb.countplot(x="label", hue="pos_or_neg", data=df)
# ax.set_xlabel("Comparison between sentiment scores of fake news and real news")
# ax.set_xticklabels(['Real News', 'Fake News'])
# ax.legend(title="Classification")

# plt.show()

pd.set_option('display.max_columns', None)
# print(title_length, num_of_words_title, mean_word_length_title)
print("New:\n", df.describe(include="all"))

# Drop missing values
df.dropna(subset="text")


# Removing HTML
def remove_html(text):
    soup = BeautifulSoup(text, "lxml")
    html_free = soup.get_text()
    return html_free


# Remove punctuation
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct


# Tokenize
def tokenizer(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text.lower())


# Remove stopwords
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    return [s for s in text if s not in stop_words]


# Lemmatize
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(l) for l in text]


# Part-of-Speech Analysis
def pos_tagging(text):
    return nltk.pos_tag(text)


##drop all entries with languages other than en (70700 entries), ru (156), es (147), de (112) and fr (47)
df[(df.language == "en") & (df.language == "ru") & (df.language == "fr") & (df.language == "de") & (df.language == "es")]

##remove links in strings
for i in range(0,len(df.index)):
    test = df.loc[i, "text"]
    df.loc[i, "text"] = re.sub(r'http\S+', '', str(df.loc[i,"text"]))
    # if test != df.loc[i,"text"]:
    #     print("Before:", test)
    #     print("After:", df.loc[i,"text"])
    df.loc[i, "title"] = re.sub(r'http\S+', '', str(df.loc[i, "title"]))
#Before: A person's skin, ancestry, and bank balance have nothing to do with their intrinsic value. https://t.co/5JsyVAKQRL  Ben Sasse (@BenSasse) September 28, 20176
#After: A person's skin, ancestry, and bank balance have nothing to do with their intrinsic value.   Ben Sasse (@BenSasse) September 28, 20176

#Before:
# First they ignore you, then they laugh at you, then they fight you, then you win. // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ <span class="mceItemHidden" data-mce-bogus="1"><span></span>&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;(function(d, s, id) {  var js, <span class="mceItemHidden" data-mce-bogus="1"><span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span></span> = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&amp;version=v2.3";  <span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span>.parentNode.insertBefore(js, fjs);}(document, 'script', '<span class="hiddenSpellError" pre="" data-mce-bogus="1">facebook-jssdk</span>')); // ]]&gt;Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.
#After:
# First they ignore you, then they laugh at you, then they fight you, then you win. (function(d, s, id) {  var js, fjs = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.3";  fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk')); // ]]>Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.
#ideally we can also remove js elements


df_preprocessed = df
df_preprocessed["text"] = df["text"].apply(lambda x: remove_html(x))  # Remove html
#Before:
# First they ignore you, then they laugh at you, then they fight you, then you win. // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ <span class="mceItemHidden" data-mce-bogus="1"><span></span>&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;(function(d, s, id) {  var js, <span class="mceItemHidden" data-mce-bogus="1"><span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span></span> = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&amp;version=v2.3";  <span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span>.parentNode.insertBefore(js, fjs);}(document, 'script', '<span class="hiddenSpellError" pre="" data-mce-bogus="1">facebook-jssdk</span>')); // ]]&gt;Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.
#After:
# First they ignore you, then they laugh at you, then they fight you, then you win. (function(d, s, id) {  var js, fjs = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.3";  fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk')); // ]]>Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.
#ideally we could also remove js elements
df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: remove_punctuation(x))  # Remove punctuation

##tf idf extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
text = []
for i in range(0,len(df.index)):
    text.append(df.loc[i,"text"])
matrix = vectorizer.fit_transform(text)
pd.DataFrame(matrix.toarray())
print(vectorizer.get_feature_names())
pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())

##repeat for titles
df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: tokenizer(x))  # tokenization
#Before: No comment is expected from Barack Obama Members of the ...
#After: 'no', 'comment', 'is', 'expected', 'from', 'barack', 'obama', 'members', 'of', 'the'
df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: remove_stopwords(x))  # Remove stop words
df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: lemmatize_text(x))  # Lemmatize
df_preprocessed["pos_tagged_text"] = df_preprocessed["text"].apply(lambda x: pos_tagging(x))  # POS-tagging

df_preprocessed["title"] = df["title"].apply(lambda x: remove_html(x))  # Remove html
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: remove_punctuation(x))  # Remove punctuation
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: tokenizer(x))  # Remove tokenization
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: remove_stopwords(x))  # Remove stop words
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: lemmatize_text(x))  # Lemmatize
df_preprocessed["pos_tagged_title"] = df_preprocessed["title"].apply(lambda x: pos_tagging(x))  # POS-tagging


##tf idf extraction



print(df_preprocessed.head())
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from langdetect import detect
from bs4 import BeautifulSoup
import lxml
import re
import nltk
#nltk.download('stopwords')
#nltk.download("wordnet")
#nltk.download('omw-1.4')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import os
from langdetect import DetectorFactory


###defining some functions for later

def remove_html(text):
    soup = BeautifulSoup(text, "lxml")
    html_free = soup.get_text()
    return html_free

def word_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(i) for i in text]
    return lemmatized_text

def stemming(text):
    snowballstemmer = SnowballStemmer("english")
    stemWords = [snowballstemmer.stem(i) for i in text]
    return stemWords

# # Drop missing values
# # df.dropna(subset="text")
#
#
# # New, preprocessed df
# df_preprocessed = df.remove_html(text).remove_punctuation(text)
# print(df_preprocessed.head())


#cwd = os.getcwd()
#os.chdir("C:/Users/Carlo/Desktop/UNI/Applied Machine Learning/datasets")
#df = pd.read_csv("datasets/WELFAKE_Dataset.csv", sep=",", low_memory=False)
df = pd.read_csv("WELFAKE_Dataset_modified.csv", sep=",", low_memory=False, nrows=1000)

##only run once to compute language feature (en, fr etc.)
# DetectorFactory.seed = 0
# language_list =[]
# for i in range(0,len(df.index)):
#     try:
#         language = detect(df.loc[i, "text"])
#         if language == "":
#             language = "error"
#     except:
#         language = "unknown"
#     language_list.append(language)
# print(len(language_list))
# df["language"] = language_list
# df.to_csv("WELFAKE_Dataset_modified.csv", index=False)
#
# print(df["language"].value_counts())
# for i in range(0,len(df.index)):
#     if df.loc[i, "language"] == "unknown":
#         print(df.loc[i, "text"])

##generate Wordclouds
# wc = WordCloud().generate(df.groupby('label')['title'].sum()[1])
# plt.figure(figsize=(15,15))
# plt.imshow(wc, interpolation='bilinear')
# plt.axis("off")
# plt.show()

##gathering information on dataset
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

#Column 0 = Index (int)
#Column 1 = title (string)
#Column 2 = text (string)
#Column 3 = labeled news as fake or real (fake = 1, real = 0)


#There are 37106 fake articles and 35028 real articles in the dataset
print("Title null entries")
print(df.title.isnull().sum())
#There are 558 entries without a title
print("Text null entries")
print(df.text.isnull().sum())
#And 39 entries without text
#df1 = df[df.isna().any(axis=1)] #saves all entries with no title and/or no text in df1





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
#Befor: A person's skin, ancestry, and bank balance have nothing to do with their intrinsic value. https://t.co/5JsyVAKQRL  Ben Sasse (@BenSasse) September 28, 20176
#After: A person's skin, ancestry, and bank balance have nothing to do with their intrinsic value.   Ben Sasse (@BenSasse) September 28, 20176

##remove html from dataframe
for i in range(0,len(df.index)):
    soup_text = BeautifulSoup(str(df.text.iloc[i]), "lxml")
    html_free = soup_text.get_text()
    df.loc[i, "text"] = html_free
    soup_title = BeautifulSoup(str(df.title.iloc[i]), "lxml")
    html_free = soup_title.get_text()
    df.loc[i, "title"] = html_free
#Before:
# First they ignore you, then they laugh at you, then they fight you, then you win. // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ <span class="mceItemHidden" data-mce-bogus="1"><span></span>&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;(function(d, s, id) {  var js, <span class="mceItemHidden" data-mce-bogus="1"><span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span></span> = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&amp;version=v2.3";  <span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span>.parentNode.insertBefore(js, fjs);}(document, 'script', '<span class="hiddenSpellError" pre="" data-mce-bogus="1">facebook-jssdk</span>')); // ]]&gt;Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.
#After:
# First they ignore you, then they laugh at you, then they fight you, then you win. (function(d, s, id) {  var js, fjs = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.3";  fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk')); // ]]>Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.
#Generated using:
    # if len(df.loc[i, "text"]) < len(test)-5:
    #     print("Mismatch:",test)
    #     print("new version:", df.loc[i, "text"])

##Tokenize text and titles (and lowercasing)
tokenizer = RegexpTokenizer(r'\w+')
print("Before Tokenization:")
print(df["text"].head(10))
df["text"] = df["text"].apply(lambda x: tokenizer.tokenize(x.lower()))
df["title"] = df["title"].apply(lambda x: tokenizer.tokenize(x.lower()))
print("After Tokenization:")
print(df["text"].head(10))
#Before: No comment is expected from Barack Obama Members of the ...
#After: 'no', 'comment', 'is', 'expected', 'from', 'barack', 'obama', 'members', 'of', 'the'


##Removal of stopwords e.g. I, me, we, you etc.
#Approach 1 (less efficient, but no annoying '' lists)
def stopwords_removal (text):
    words = [i for i in text if i not in stopwords.words("english")]
    return words

df["text"] = df["text"].apply(lambda x: stopwords_removal(x))
print("After Stopwordremoval:")
print(df["text"].head(10))

#Approach 2 (way more efficient)
# pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
# print("Before Stopwordremoval:", df["text"].head(10))
# for i in range(0,len(df.index)):
#     df.loc[i, "text"] = pattern.sub('', str(df.loc[i, "text"]))
#     df.loc[i, "title"] = pattern.sub('', str(df.loc[i, "title"]))
# print("After Stopwordremoval:", df["text"].head(10))


##Word Lemmatization (standardizes word from different declinations etc. to their base form). a more aggressive approach would be "stemming"
# df["text"] = df["text"].apply(lambda x: word_lemmatizer(x))
# df["title"] = df["title"].apply(lambda x: word_lemmatizer(x))
# print("After Lemmatization:")
# print(df["text"].head(10))

#unfortunately it seems that runtime is too long for this
##second approach "stemming"
df["text"] = df["text"].apply(lambda x: stemming(x))
df["text"] = df["text"].apply(lambda x: stemming(x))
print("After Stemming:")
print(df["text"].head(10))


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




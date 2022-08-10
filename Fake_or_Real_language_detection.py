from typing import Any
from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import mean, std
import seaborn as sb
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import os
from bs4 import BeautifulSoup
import string
import lxml
from lxml.html import fromstring
from sklearn import preprocessing, model_selection, naive_bayes, __all__
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

# cwd = os.getcwd()
os.chdir(r"C:\Users\Jens\Desktop\Unizeug\Master\2. Semester\Applied Machine Learning\Final project")
#os.chdir(r"C:\Users\D\Desktop\PycharmProjects\PVAÜbung\PVAProjekt")
df = pd.read_csv("WELFAKE_Dataset_modified_nrow=10000.csv", sep=",", low_memory=False, nrows=100)
#df.to_csv("WELFAKE_Dataset_modified_nrow=10000.csv", sep=",")
# df = pd.read_csv("WELFAKE_Dataset_modified.csv", sep=",", low_memory=False)

#print(df.label.value_counts())

# There are 37106 fake articles and 35028 real articles in the dataset
#print("Title null entries")
#print(df.title.isnull().sum())

# There are 558 entries without a title
#print("Text null entries")
#print(df.text.isnull().sum())

# And 39 entries without text


# adds title_length and text_length to dataframe
title_length = []
text_length = []
for i in range(0, len(df.index)):
    title_length.append(len(str(df.loc[i, "title"])))
    text_length.append(len(str(df.loc[i, "text"])))
df["title_length"] = title_length
df["text_length"] = text_length


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

for i in range(len(df.index)):
    if df.loc[i, "title_wordnum"] >= 60:
        print(df.loc[i, "title"])

# Plot of text/title length and text/title mean word length in one subplot
#fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#sb.histplot(ax=axes[1, 0], data=df, x="text_length", kde=True)
#axes[1, 0].set_xlim(0, 30000)
#axes[1, 0].set_xlabel("Length of Text [characters]")
#sb.histplot(ax=axes[0, 0], data=df, x="title_length", kde=True)
#axes[0, 0].set_xlim(0, )
#axes[0, 0].set_xlabel("Length of Title [characters]")
#sb.histplot(ax=axes[1, 1], data=df, x="text_meanlen", kde=True)
#axes[1, 1].set_xlim(0, 25)
#axes[1, 1].set_xlabel("Mean word length [text]")
#sb.histplot(ax=axes[0, 1], data=df, x="title_meanlen", kde=True)
#axes[0, 1].set_xlim(0, 25)
#axes[0, 1].set_xlabel("Mean word length [title]")

# plt.show()

df_fake = df.loc[df["label"] == 1]
df_real = df.loc[df["label"] == 0]
# Plotting of differences in title length and mean word length by Fake/Real News categorization
#fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#sb.histplot(ax=axes[0, 0], data=df_real, x="title_wordnum", kde=True)
#axes[0, 0].set_xlim(0, )
#axes[0, 0].set_ylim(0, )
#axes[0, 0].set_xlabel("Length of Title [words]")
#axes[0, 0].set_title("Real News")
#sb.histplot(ax=axes[0, 1], data=df_fake, x="title_wordnum", kde=True)
#axes[0, 1].set_xlim(0, )
#axes[0, 1].set_ylim(0, )
#axes[0, 1].set_xlabel("Length of Title [words]")
#axes[0, 1].set_title("Fake News")
#sb.histplot(ax=axes[1, 0], data=df_real, x="title_meanlen", kde=True)
#axes[1, 0].set_xlim(0, 15)
#axes[1, 0].set_ylim(0, 1500)
#axes[1, 0].set_xlabel("Mean word length [title]")
#sb.histplot(ax=axes[1, 1], data=df_fake, x="title_meanlen", kde=True)
#axes[1, 1].set_xlim(0, 15)
#axes[1, 1].set_ylim(0, 1500)
#axes[1, 1].set_xlabel("Mean word length [title]")

# plt.show()

# Plotting of differences in text length and mean word length by Fake/Real News categorization

#fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#sb.histplot(ax=axes[0, 0], data=df_real, x="title_wordnum", kde=True)
#axes[0, 0].set_xlim(0, )
#axes[0, 0].set_ylim(0, )
#axes[0, 0].set_xlabel("Length of text [words]")
#axes[0, 0].set_title("Real News")
#sb.histplot(ax=axes[0, 1], data=df_fake, x="title_wordnum", kde=True)
#axes[0, 1].set_xlim(0, )
#axes[0, 1].set_ylim(0, )
#axes[0, 1].set_xlabel("Length of text [words]")
#axes[0, 1].set_title("Fake News")
#sb.histplot(ax=axes[1, 0], data=df_real, x="text_meanlen", kde=True)
#axes[1, 0].set_xlim(0, 15)
#axes[1, 0].set_ylim(0, 1300)
#axes[1, 0].set_xlabel("Mean word length [text]")
#sb.histplot(ax=axes[1, 1], data=df_fake, x="text_meanlen", kde=True)
#axes[1, 1].set_xlim(0, 15)
#axes[1, 1].set_ylim(0, 1300)
#axes[1, 1].set_xlabel("Mean word length [text]")

# plt.show()


# print(title_length, num_of_words_title, mean_word_length_title)
#print("New:\n", df.describe(include="all"))

# Drop missing values
df["text"].dropna()


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
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(l) for l in text]


# Part-of-Speech Analysis
def pos_tagging(text):
    return nltk.pos_tag(text)


# Sentiment Analysis
def sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    # add check for "language" variable here.
    sentiment_dict = sia.polarity_scores(text)
    return sentiment_dict['compound']


# Standardizer
def standardize(values):
    scaler = preprocessing.scale(values)
    return scaler

def crossval(clf, x_train, y):
    scores = cross_validate(clf, x_train, y)
    print("\n Cross Validation Scores ->", scores["test_score"])
    print("Mean ->", mean(scores["test_score"]))
    print("Standard Deviation ->", std(scores["test_score"]), "\n")


def we_will_roc_you(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()


##drop all entries with languages other than en (70700 entries), ru (156), es (147), de (112) and fr (47)
# df[(df.language == "en") & (df.language == "ru") & (df.language == "fr") & (df.language == "de") & (df.language == "es")]
#df[(df.language == "en")]

##remove links in strings
for i in range(0, len(df.index)):
    test = df.loc[i, "text"]
    df.loc[i, "text"] = re.sub(r'http\S+', '', str(df.loc[i, "text"]))
    df.loc[i, "title"] = re.sub(r'http\S+', '', str(df.loc[i, "title"]))
# Before: A person's skin, ancestry, and bank balance have nothing to do with their intrinsic value. https://t.co/5JsyVAKQRL  Ben Sasse (@BenSasse) September 28, 20176
# After: A person's skin, ancestry, and bank balance have nothing to do with their intrinsic value.   Ben Sasse (@BenSasse) September 28, 20176


df_preprocessed = df
df_preprocessed["text"] = df["text"].apply(lambda x: remove_html(x))  # Remove html
# Before:
# First they ignore you, then they laugh at you, then they fight you, then you win. // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ // < ![CDATA[ <span class="mceItemHidden" data-mce-bogus="1"><span></span>&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;&lt;span&gt;&lt;/span&gt;(function(d, s, id) {  var js, <span class="mceItemHidden" data-mce-bogus="1"><span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span></span> = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&amp;version=v2.3";  <span class="hiddenSpellError" pre="" data-mce-bogus="1">fjs</span>.parentNode.insertBefore(js, fjs);}(document, 'script', '<span class="hiddenSpellError" pre="" data-mce-bogus="1">facebook-jssdk</span>')); // ]]&gt;Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.
# After:
# First they ignore you, then they laugh at you, then they fight you, then you win. (function(d, s, id) {  var js, fjs = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.3";  fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk')); // ]]>Posted by Sarah Palin on Wednesday, February 24, 2016This quote has long been used by civil rights pioneers in their quest for justice and equality.

##js removal

for i in range(0, len(df.index)):
    try:
        df_preprocessed.loc[i, "text"] = fromstring(df_preprocessed.loc[i, "text"]).text_content()
    except lxml.etree.ParserError:
        df_preprocessed.loc[i, "title"] = df_preprocessed.loc[i, "title"]
# Before: This is how he ll be remembered:// <![CDATA[ (function(d, s, id) { var js, fjs = d.getElementsByTagName(s)[0]; if (d.getElementById(id)) return; js = d.createElement(s); js.id = id; js.src = "//connect.facebook.net/en_GB/sdk.js#xfbml=1&version=v2.3"; fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk')); // ]]>WATCH: Protests erupted in Chicago Tuesday night in the wake of first-degree
# After: This is how he ll be remembered:// WATCH: Protests erupted in Chicago Tuesday night in the wake of first-degree

df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: remove_punctuation(x))  # Remove punctuation
df_preprocessed["sentiment_score_text"] = df["text"].apply(lambda x: sentiment_score(x))

full_text = ""
for i in range(0, len(df_preprocessed.index)):
    full_text = full_text + df_preprocessed.loc[i, "text"]

# TFIDF DER HURENSOHN

PETER = df_preprocessed["text"]

#for i in range(0,len(PETER.index)):
    #PETER.iloc[i] = PETER.iloc[i] + df_preprocessed.loc[i, "title"]

#print("hier ist z!",z)
df_preprocessed["sentiment_score_title"] = df_preprocessed["title"].apply(lambda x: sentiment_score(x))

v = TfidfVectorizer()
tfidf = []
for i in range(0,len(PETER.index)):
    tfidf.append(PETER.iloc[i])

x = v.fit_transform(tfidf)
print(x.shape)
z = x.toarray()
print(z.shape)


# TF IDF SECTION
X_train_withtfidf, X_test_withtfidf, y_train_withtfidf, y_test_withtfidf = train_test_split(df_preprocessed["text"], df_preprocessed['label'], test_size=0.3, random_state=420)


Encoder = LabelEncoder()
y_train_withtfidf = Encoder.fit_transform(y_train_withtfidf)
y_test_withtfidf = Encoder.fit_transform(y_test_withtfidf)

X_train_withtfidf = v.transform(X_train_withtfidf).toarray()
X_test_withtfidf = v.transform(X_test_withtfidf).toarray()

# #NB_clf
Naive2 = naive_bayes.GaussianNB()
Naive2.fit(X_train_withtfidf,y_train_withtfidf)# predict the labels on validation dataset
predictions_NB = Naive2.predict(X_test_withtfidf)# Use accuracy_score function to get the accuracy
print("Naive Bayes tfidf Accuracy Score -> ",accuracy_score(predictions_NB, y_test_withtfidf)*100)
print("Naive Bayes tfidf Precision Score -> ", precision_score(predictions_NB, y_test_withtfidf)*100)
print("Naive Bayes tfidf Recall Score -> ", recall_score(predictions_NB, y_test_withtfidf)*100)
print("Naive Bayes tfidf F1-Score -> ", f1_score(predictions_NB, y_test_withtfidf)*100)
print(confusion_matrix(predictions_NB, y_test_withtfidf))


crossval(naive_bayes.GaussianNB(), z, df_preprocessed['label'])
roc_pred_tfidf_NB = Naive2.fit(X_train_withtfidf, y_train_withtfidf).predict_proba(X_test_withtfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withtfidf,  roc_pred_tfidf_NB)
we_will_roc_you(fpr, tpr)
plt.savefig("nb_tfidf_roc.jpg")
plt.close()

#SVC()
clf_svc = SVC(probability=True)
clf_svc.fit(X_train_withtfidf,y_train_withtfidf)
predictions_SVM = clf_svc.predict(X_test_withtfidf)
print("Support Vector Machine tfidf Accuracy Score -> ", accuracy_score(predictions_SVM, y_test_withtfidf)*100)
print("Support Vector Machine tfidf Precision Score -> ", precision_score(predictions_SVM, y_test_withtfidf)*100)
print("Support Vector Machine tfidf Recall Score -> ", recall_score(predictions_SVM, y_test_withtfidf)*100)
print("Support Vector Machine tfidf F1-Score -> ", f1_score(predictions_SVM, y_test_withtfidf)*100)
print(confusion_matrix(predictions_SVM, y_test_withtfidf))

crossval(SVC(), z, df_preprocessed['label'])
roc_pred_tfidf_SVC = clf_svc.fit(X_train_withtfidf, y_train_withtfidf).predict_proba(X_test_withtfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withtfidf,  roc_pred_tfidf_SVC)
we_will_roc_you(fpr, tpr)
plt.savefig("svc_tfidf_roc.jpg")
plt.close()

# Random Forest
clf_rf = RandomForestClassifier()
clf_rf = clf_rf.fit(X_train_withtfidf,y_train_withtfidf)
predictions_rf = clf_rf.predict(X_test_withtfidf)

cm2 = confusion_matrix(y_test_withtfidf, predictions_rf)
print("Random Forest tfidf Accuracy Score -> ", accuracy_score(predictions_rf, y_test_withtfidf)*100)
print("Random Forest tfidf Precision Score -> ", precision_score(predictions_rf, y_test_withtfidf)*100)
print("Random Forest tfidf Recall Score -> ", recall_score(predictions_rf, y_test_withtfidf)*100)
print("Random Forest tfidf F1-Score -> ", f1_score(predictions_rf, y_test_withtfidf)*100)
print(cm2)

crossval(RandomForestClassifier(), z, df_preprocessed['label'])
roc_pred_tfidf_SVC = clf_rf.fit(X_train_withtfidf, y_train_withtfidf).predict_proba(X_test_withtfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withtfidf,  roc_pred_tfidf_SVC)
we_will_roc_you(fpr, tpr)
plt.savefig("rf_tfidf_roc.jpg")
plt.close()


# Decision Tree
from sklearn import tree
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(X_train_withtfidf, y_train_withtfidf)
y_pred_dt = clf_dt.predict(X_test_withtfidf)

cm3 = confusion_matrix(y_pred_dt, y_test_withtfidf)
print("Decision Tree tfidf Accuracy Score -> ", accuracy_score(y_pred_dt, y_test_withtfidf)*100)
print("Decision Tree tfidf Precision Score -> ", precision_score(y_pred_dt, y_test_withtfidf)*100)
print("Decision Tree tfidf Recall Score -> ", recall_score(y_pred_dt, y_test_withtfidf)*100)
print("Decision Tree tfidf Tree F1-Score -> ", f1_score(y_pred_dt, y_test_withtfidf)*100)
print(cm3)

crossval(tree.DecisionTreeClassifier(), z, df_preprocessed['label'])
roc_pred_tfidf_dt = clf_dt.fit(X_train_withtfidf, y_train_withtfidf).predict_proba(X_test_withtfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withtfidf,  roc_pred_tfidf_dt)
we_will_roc_you(fpr, tpr)
plt.savefig("dt_tfidf_roc.jpg")
plt.close()


#K-Nearest-Neighbors
from sklearn import neighbors
clf_nbrs = neighbors.KNeighborsClassifier()
clf_nbrs.fit(X_train_withtfidf, y_train_withtfidf)
y_pred_nbrs = clf_nbrs.predict(X_test_withtfidf)
cm4 = confusion_matrix(y_pred_nbrs, y_test_withtfidf)

print("NeighborsClassifier tfidf Accuracy Score -> ", accuracy_score(y_pred_nbrs, y_test_withtfidf)*100)
print("NeighborsClassifier tfidf Precision Score -> ", precision_score(y_pred_nbrs, y_test_withtfidf)*100)
print("NeighborsClassifier tfidf Recall Score -> ", recall_score(y_pred_nbrs, y_test_withtfidf)*100)
print("NeighborsClassifier tfidf F1-Score -> ", f1_score(y_pred_nbrs, y_test_withtfidf)*100)
print(cm4)

crossval(neighbors.KNeighborsClassifier(), z, df_preprocessed['label'])
roc_pred_tfidf_knn = clf_nbrs.fit(X_train_withtfidf, y_train_withtfidf).predict_proba(X_test_withtfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withtfidf,  roc_pred_tfidf_dt)
we_will_roc_you(fpr, tpr)
plt.savefig("knn_tfidf_roc.jpg")
plt.close()

df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: tokenizer(x))  # tokenization
df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: remove_stopwords(x))  # Remove stop words
df_preprocessed["pos_tagged_text"] = df_preprocessed["text"].apply(lambda x: pos_tagging(x))  # POS-tagging
# Before: No comment is expected from Barack Obama Members of the ...
# After: 'no', 'comment', 'is', 'expected', 'from', 'barack', 'obama', 'members', 'of', 'the'
df_preprocessed["text"] = df_preprocessed["text"].apply(lambda x: lemmatize(x))  # Lemmatize

##repeat for titles
df_preprocessed["title"] = df["title"].apply(lambda x: remove_html(x))  # Remove html
for i in range(0, len(df.index)):
    try:
        df_preprocessed.loc[i, "title"] = fromstring(df_preprocessed.loc[i, "title"]).text_content()
    except lxml.etree.ParserError:
        df_preprocessed.loc[i, "title"] = df_preprocessed.loc[i, "title"]
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: remove_punctuation(x))  # Remove punctuation
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: tokenizer(x))  # Remove tokenization
df_preprocessed["pos_tagged_title"] = df_preprocessed["title"].apply(lambda x: pos_tagging(x))  # POS-tagging
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: remove_stopwords(x))  # Remove stop words
df_preprocessed["title"] = df_preprocessed["title"].apply(lambda x: lemmatize(x))  # Lemmatize

# Standardize continuous data
df_preprocessed["title_meanlen_standardized"] = \
    standardize(df_preprocessed["title_meanlen"])  # Standardize mean length of title
df_preprocessed["text_meanlen_standardized"] = \
    standardize(df_preprocessed["text_meanlen"])  # Standardize mean length of text
df_preprocessed["title_wordnum_standardized"] = \
    standardize(df_preprocessed["title_wordnum"])  # Standardize word count of title
df_preprocessed["text_wordnum_standardized"] = \
    standardize(df_preprocessed["text_wordnum"])  # Standardize word count of text

# Get rid of the index column
df_preprocessed: Any = df_preprocessed.drop('Unnamed: 0', axis=1)

# Let's check how well the data is balanced between fake and real news and plot the respective sentiment scores
fake_count = len(df_fake) / df.shape[0]
real_count = len(df_real) / df.shape[0]

label_count = [fake_count, real_count]

#df["pos_or_neg"] = df["sentiment score_text"].apply(
#    lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))
#fig = plt.figure(figsize=(18, 10))
#ax1 = plt.subplot2grid((1, 2), (0, 0))
#plt.pie(x=label_count, explode=[0.1, 0.1], colors=['firebrick', 'navy'], startangle=90, shadow=True,
#        labels=['Fake News', 'True News'], autopct='%1.1f%%')

#ax1 = plt.subplot2grid((1, 2), (0, 1))
#ax = sb.countplot(x="label", hue="pos_or_neg", data=df)
#ax.set_xlabel("Comparison between sentiment scores of fake news and real news")
#ax.set_xticklabels(['Real News', 'Fake News'])
#plt.title("Classification")

# plt.show()

#pd.set_option('display.max_columns', None)
#
#print(df_preprocessed.head())

# Random Forest
#print("RAndomforest")
#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df_preprocessed['text'],df_preprocessed['label'],test_size=0.3)
#Encoder = LabelEncoder()
#Train_Y = Encoder.fit_transform(Train_Y)
#Test_Y = Encoder.fit_transform(Test_Y)
#print(Test_X)

#clf = RandomForestClassifier(n_estimators =400,criterion="entropy")
#clf = clf.fit(Train_X, Train_Y)
#scores = cross_val_score(clf, Test_X, Test_Y>, cv=5)
#y_pred = clf.predict(Test_X)
#from sklearn.metrics import confusion_matrix
#cm2 = confusion_matrix(Test_Y, y_pred)
#print("hier!", y_pred, cm2)

X_withouttfidf = df_preprocessed.loc[:, ["title_meanlen_standardized", "text_meanlen_standardized", "title_wordnum_standardized", "text_wordnum_standardized", "sentiment_score_text", "sentiment_score_title"]]
X_train_withouttfidf, X_test_withouttfidf, y_train_withouttfidf, y_test_withouttfidf = train_test_split(X_withouttfidf, df_preprocessed['label'], test_size=0.3, random_state=420)

# SVC
clf = SVC(probability=True)
clf.fit(X_train_withouttfidf, y_train_withouttfidf)
predictions_SVM = clf.predict(X_test_withouttfidf)

print("Support Vector Machine Accuracy Score -> ", accuracy_score(predictions_SVM, y_test_withouttfidf)*100)
print("Support Vector Machine Precision Score -> ", precision_score(predictions_SVM, y_test_withouttfidf)*100)
print("Support Vector Machine Recall Score -> ", recall_score(predictions_SVM, y_test_withouttfidf)*100)
print("Support Vector Machine F1-Score -> ", f1_score(predictions_SVM, y_test_withouttfidf)*100)

crossval(SVC(), X_withouttfidf, df_preprocessed['label'])
roc_pred_svc = clf.fit(X_train_withouttfidf, y_train_withouttfidf).predict_proba(X_test_withouttfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withouttfidf,  roc_pred_svc)
we_will_roc_you(fpr, tpr)
plt.savefig("svc_roc.jpg")
plt.close()

#NB classifier
Naive = naive_bayes.GaussianNB()
Naive.fit(X_train_withouttfidf, y_train_withouttfidf)# predict the labels on validation dataset
predictions_NB = Naive.predict(X_test_withouttfidf)# Use accuracy_score function to get the accuracy
print("Gaussian Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, y_test_withouttfidf)*100)
print("Gaussian Naive Bayes Precision Score -> ", precision_score(predictions_NB, y_test_withouttfidf)*100)
print("Gaussian Naive Bayes Recall Score -> ", recall_score(predictions_NB, y_test_withouttfidf)*100)
print("Gaussian Naive Bayes F1-Score -> ", f1_score(predictions_NB, y_test_withouttfidf)*100)

print(confusion_matrix(predictions_NB, y_test_withouttfidf))

crossval(naive_bayes.GaussianNB(), X_withouttfidf, df_preprocessed['label'])
roc_pred_nb = Naive.fit(X_train_withouttfidf, y_train_withouttfidf).predict_proba(X_test_withouttfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withouttfidf,  roc_pred_nb)
we_will_roc_you(fpr, tpr)
plt.savefig("nb_roc.jpg")
plt.close()


#Random Forest
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train_withouttfidf, y_train_withouttfidf)
#scores = cross_val_score(clf, Test_X, Test_Y>, cv=5)
y_pred = clf_rf.predict(X_test_withouttfidf)
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_pred, y_test_withouttfidf)

print("Random Forest Accuracy Score -> ", accuracy_score(y_pred, y_test_withouttfidf)*100)
print("Random Forest Precision Score -> ", precision_score(y_pred, y_test_withouttfidf)*100)
print("Random Forest Recall Score -> ", recall_score(y_pred, y_test_withouttfidf)*100)
print("Random Forest F1-Score -> ", f1_score(y_pred, y_test_withouttfidf)*100)
print(cm2)

crossval(RandomForestClassifier(), X_withouttfidf, df_preprocessed['label'])
roc_pred_rf = clf_rf.fit(X_train_withouttfidf, y_train_withouttfidf).predict_proba(X_test_withouttfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withouttfidf,  roc_pred_rf)
we_will_roc_you(fpr, tpr)
plt.savefig("rf_roc.jpg")
plt.close()

# Decision Tree
from sklearn import tree
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(X_train_withouttfidf, y_train_withouttfidf)
y_pred_dt = clf_dt.predict(X_test_withouttfidf)
cm3 = confusion_matrix(y_pred_dt, y_test_withouttfidf)

print("Decision Tree Accuracy Score -> ", accuracy_score(y_pred_dt, y_test_withouttfidf)*100)
print("Decision Tree Precision Score -> ", precision_score(y_pred_dt, y_test_withouttfidf)*100)
print("Decision Tree Recall Score -> ", recall_score(y_pred_dt, y_test_withouttfidf)*100)
print("Decision Tree F1-Score -> ", f1_score(y_pred_dt, y_test_withouttfidf)*100)
print(cm3)

crossval(tree.DecisionTreeClassifier(), X_withouttfidf, df_preprocessed['label'])
roc_pred_dt = clf_dt.fit(X_train_withouttfidf, y_train_withouttfidf).predict_proba(X_test_withouttfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withouttfidf,  roc_pred_dt)
we_will_roc_you(fpr, tpr)
plt.savefig("dt_roc.jpg")
plt.close()


#K-Nearest-Neighbors
from sklearn import neighbors
clf_nbrs = neighbors.KNeighborsClassifier()
clf_nbrs.fit(X_train_withouttfidf, y_train_withouttfidf)
y_pred_nbrs = clf_nbrs.predict(X_test_withouttfidf)
cm4 = confusion_matrix(y_pred_nbrs, y_test_withouttfidf)

print("NeighborsClassifier Accuracy Score -> ", accuracy_score(y_pred_dt, y_test_withouttfidf)*100)
print("NeighborsClassifier Precision Score -> ", precision_score(y_pred_dt, y_test_withouttfidf)*100)
print("NeighborsClassifier Recall Score -> ", recall_score(y_pred_dt, y_test_withouttfidf)*100)
print("NeighborsClassifier F1-Score -> ", f1_score(y_pred_dt, y_test_withouttfidf)*100)
print(cm4)

crossval(neighbors.KNeighborsClassifier(), X_withouttfidf, df_preprocessed['label'])
roc_pred_knn = clf_nbrs.fit(X_train_withouttfidf, y_train_withouttfidf).predict_proba(X_test_withouttfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test_withouttfidf,  roc_pred_knn)
we_will_roc_you(fpr, tpr)
plt.savefig("knn_roc.jpg")
plt.close()
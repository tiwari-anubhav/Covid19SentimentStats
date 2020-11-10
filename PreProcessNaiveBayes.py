from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from operator import mul
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # or any other NB model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


file = pd.read_csv("Data/TrainingData/Tweets_with_Social_Distancing_All_train.csv", index_col=0)
file.clean_tweet=file.clean_tweet.astype(str)
# file.to_csv("processed_labeled.csv")

x = file['clean_tweet']
y = file['polarity_values']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
count_vect = CountVectorizer()
train_x_counts = count_vect.fit_transform(train_x)
test_x_counts = count_vect.transform(test_x)
# tf_transformer = TfidfTransformer(use_idf=False).fit(train_x_counts)
# train_x_tf = tf_transformer.transform(train_x_counts)
# print(train_x_tf.shape)
# print(test_x_counts.shape)


#########################################
# print(train_x_tf)
nb_classifier = MultinomialNB()
# train_y = np.array(train_y.unique())
# test_y = np.array(test_y.unique())
nb_classifier.fit(train_x_counts, train_y)

y_pred = nb_classifier.predict(test_x_counts)

acc_score = accuracy_score(test_y, y_pred)
conf_mat = confusion_matrix(test_y, y_pred, labels = [0, 1, 2])
print(acc_score)
print(conf_mat)
target_names = ['Negative', 'Neutral', 'Positive']
print(classification_report(test_y, y_pred, target_names=target_names))


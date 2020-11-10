from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def highestSVM(x):
    all_accuracies = {}
    for item in x:
        svm = item.fit(train_x,train_y)
        y_pred = svm.predict(test_x)
        all_accuracies[item] = accuracy_score(test_y, y_pred)
    max_accuracy = max(all_accuracies, key=all_accuracies.get)
    score = all_accuracies[max_accuracy]
    return max_accuracy, score




file = pd.read_csv("Data/TrainingData/Tweets_with_mask_All_train.csv", index_col=0)
file.clean_tweet=file.clean_tweet.astype(str)
# file.to_csv("processed_labeled.csv")
x = file['clean_tweet']
y = file['polarity_values']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
# train_x_counts = count_vect.fit_transform(train_x)
# test_x_counts = count_vect.transform(test_x)
object_list = []
svm0 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1))])
object_list.append(svm0)
svm01 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1, gamma = 0.001))])
object_list.append(svm01)
svm1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1))])
object_list.append(svm1)
svm11 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1, gamma=0.001))])
object_list.append(svm11)
svm2 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=10))])
object_list.append(svm2)
svm21 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=10, gamma=0.001))])
object_list.append(svm21)
svm3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=10))])
object_list.append(svm3)
svm31 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=10, gamma = 0.001))])
object_list.append(svm31)
svm4 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=100))])
object_list.append(svm4)
svm41 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=100, gamma=0.001))])
object_list.append(svm41)
svm5 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=100))])
object_list.append(svm5)
svm51 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=100, gamma = 0.001))])
object_list.append(svm51)
svm6 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1000))])
object_list.append(svm6)
svm61 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1000, gamma = 0.001))])
object_list.append(svm61)
svm7 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1000))])
object_list.append(svm7)
svm71 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1000, gamma = 0.001))])
object_list.append(svm71)



bestSVM,score = highestSVM(object_list)
print(bestSVM,score)

test_file = pd.read_csv("Data/TestData/TestData_Mask.csv")
test_file["Predicted_Sentiment"] = bestSVM.predict(test_file['clean_tweet'])
test_file.to_csv("Data/TestData/TestData_Mask.csv", index = False)
#test_file.clean_tweet=test_file.clean_tweet.astype(str)
#test_file_x = test_file['clean_tweet']

# stop_words.update(STOPWORDS)
# stop_words.update(["book", "nook", "read", "n't", "kindl","camera","use","button","len","one"])
# positive_wc = WordCloud(width = 3000,
#     height = 2000,stopwords = stop_words, background_color='white').generate(str(positive))
# fig = plt.figure(figsize = (40, 30))
# plt.imshow(positive_wc, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad = 0)
# plt.show()

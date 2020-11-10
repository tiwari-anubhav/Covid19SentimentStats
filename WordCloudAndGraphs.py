import pandas as pd
import datetime
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

def get_month(x):
    try:
        d = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        d = datetime.datetime.strptime(x, "%Y-%m-%d")
    except Exception as e:
        d = None
    if d is not None:
        d = d.month
    return d

def get_stats_for_bar(stats):
    pos = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    neg = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    neut = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for k in stats.keys():
        d,s = k
        if s == 0:
            neg[d] = stats.get(k)
        if s == 1:
            neut[d] = stats.get(k)
        if s == 2:
            pos[d] = stats.get(k)

    return pos,neg,neut
stop_words = set(stopwords.words('english'))
stop_words.update(["I", "?", ".", ";", "'", ",", "[", "]", "(", ")"])
file = pd.read_csv("Data/TestData/TestData_Social_Distancing.csv")
#file = pd.read_csv("Data/TrainingData/Tweets_with_mask_All_train.csv")
file.clean_tweet=file.clean_tweet.astype(str)
print(len(file))
file['month'] = file['date'].apply(lambda x: get_month(x))
file = file[file['month'] == 9]
print(file.columns)
file = file[file['Predicted_Sentiment'] == 0]
tweet = ' '.join(list(file['clean_tweet']))


stop_words.update(["book", "nook", "read", "n't", "kindl","camera","use","button","len","one","coronavirus","virus","mask","face","masks","wearing","wear","china"])
stop_words.update(["people","like","hand","surgical","Wuhan","doctor","distancing","social distancing","social","COVID","Alone","monster","amp"])
positive_wc = WordCloud(width = 3000,
    height = 2000,stopwords = stop_words, background_color='white').generate(str(tweet))
fig = plt.figure(figsize = (40, 30))
plt.imshow(positive_wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

#################################################################################################
# Plotting Bar Charts
# file['Predicted_Sentiment'] = file['polarity_values'].apply(lambda x: x)
# file1 = pd.read_csv("Data/TestData/TestData_Mask.csv")
#
# new = pd.concat([file,file1]).reset_index(drop=True)
# req=0
# for index,row in new.iterrows():
#     if row['date'] is None or isinstance(row['date'], float):
#         req=index
#
# print(req)
# new = new.drop(index=req, axis=0)
#
#
# new['month'] = new['date'].apply(lambda x: get_month(x))
#
# labels = ['Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct']
# x = np.arange(len(labels))
# width = 0.20
#
# stats = new.groupby(['month','Predicted_Sentiment']).size()
# print(stats)
# pos, neg, neut = get_stats_for_bar(stats)
# pos = pos.values()
# neg = neg.values()
# neut = neut.values()
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width, pos, width, label='Positive')
# rects2 = ax.bar(x, neg, width, label='Negative')
# rects3 = ax.bar(x + width, neut, width, label='Neutral')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Tweets')
# ax.set_title('Tweets containing Mask by Month and Sentiment')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
#
# fig.tight_layout()
#
# plt.show()
#print(get_stats_for_bar(stats))


## Plotting Pie Charts
# labels = 'Positive','Negative','Neutral'
# sizes = [stats.get(2), stats.get(0), stats.get(1)]
# explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()

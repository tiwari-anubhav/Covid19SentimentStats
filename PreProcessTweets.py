import pandas as pd
import os,re
import preprocessor as p
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def remove_url(txt):
    #([^0-9A-Za-z \t])|(\w+:\/\/\S+)
    return " ".join(re.sub("http[s]?://\S+", "", txt).split())

def remove_hashtags(txt):
    return " ".join(re.sub(r"#(\w+)", "", txt).split())

def process_tweets(txt):
    query_text = word_tokenize(txt)
    arr = []
    for word in query_text:
        if word not in stop_words:
            arr.append(porter.stem(word))
    return arr

def select_relevant_tweets_df(df):
    if 'language' in df.columns:
        df = df[df.language == 'en']    # Only Keep English Tweets
    df['tweet'] = df['tweet'].apply(remove_url)     # Remove any Links from Tweets
    df = df[df.apply(lambda x: any(elem in x['tweet'] for elem in req_text), axis=1)]   # Check for mask in the tweet text
    df = df.drop_duplicates(subset='tweet', keep="first")   # Remove duplicate tweets
    df['hashtags'] = df['tweet'].apply(lambda x: re.findall(r"#(\w+)", x)) # Make a new column for all the hashtags
    df['clean_tweet'] = df['tweet'].apply(lambda x: remove_hashtags(x)) # Remove hashtags
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: p.clean(x)) # Clean tweet text
    df['processed_tweet'] = df['clean_tweet'].apply(lambda x: process_tweets(x))
    return df



if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.width', None)
    stop_words = set(stopwords.words('english'))
    stop_words.update(["I", "?", ".", ";", "'", ",", "[", "]", "(", ")"])
    porter = PorterStemmer()
    req_text = [' distancing']
    months = ['02-2020','03-2020','04-2020','05-2020','06-2020','07-2020','08-2020','09-2020','10-2020']
    for month in months:
        path = '../Covid19SentimentStats/Tweets/'+month
        mask_df = pd.DataFrame()
        for file in os.listdir(path):
            df = pd.read_csv(path+'/'+file)
            df = select_relevant_tweets_df(df)
            #print(len(df))
            mask_df = mask_df.append(df)
        print(len(mask_df))
        mask_df.to_csv('../Covid19SentimentStats/Processed_Tweets/'+month+'/Tweets_with_Social_Distancing.csv', index=False)

import pandas as pd
import os

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    months = []
    #months = ['02-2020','03-2020','04-2020','05-2020','06-2020','07-2020','08-2020','09-2020','10-2020']
    columns = [ 'date', 'favourites_count','followers_count', 'language', 'location', 'retweet_count','screen_name', 'tweet', 'tweet_id', 'verified', 'hashtags','clean_tweet', 'processed_tweet']
    new_df = pd.read_csv('../Covid19SentimentStats/Processed_Tweets/Tweets_with_Social_Distancing_All.csv')
    new_df = new_df.head(500)
    #new_df = pd.DataFrame()
    count = 0
    for month in months:
        path = '../Covid19SentimentStats/Processed_Tweets/'+month

        for file in os.listdir(path):
            if "Distancing" in file:
                df = pd.read_csv(path+'/'+file)
                if 'Unnamed: 0' in df.columns:
                    del df['Unnamed: 0']
                else:
                    df['language'] = 'en'
                    df['verified'] = None
                    df.rename(columns={'id': 'tweet_id', 'likes_count': 'favourites_count','retweets_count':'retweet_count','replies_count':'followers_count','geo':'location','username':'screen_name'}, inplace=True)
                    for col in df.columns:
                        if col not in columns:
                            del df[col]
                #print(df.columns)
                print(len(df))
                count = count+len(df)
                new_df = new_df.append(df)
    print(len(new_df))
    new_df = new_df.to_csv('../Covid19SentimentStats/Processed_Tweets/Tweets_with_Social_Distancing_All_train.csv', index=False)
    #new_df = new_df.head(500)
    #print(new_df.columns)
    #new_df.to_csv('../Covid19SentimentStats/Processed_Tweets/Tweets_with_mask_All_train.csv', index=False)
#[ 'date', 'favourites_count','followers_count', 'language', 'location', 'retweet_count','screen_name', 'tweet', 'tweet_id',
# 'verified', 'hashtags','clean_tweet', 'processed_tweet']
#['id', 'conversation_id', 'created_at', 'date', 'time','timezone', 'user_id', 'username', 'name', 'place', 'tweet',
# 'mentions','urls', 'photos', 'replies_count', 'retweets_count', 'likes_count','hashtags', 'cashtags', 'link', 'retweet',
# 'quote_url', 'video', 'near','geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to','retweet_date', 'translate',
# 'trans_src', 'trans_dest', 'clean_tweet','processed_tweet']

# Import the necessary methods from tweepy library
import tweepy
import pandas as pd
import os
# Enter Twitter API Keys
consumer_key = 'IUjMKkwPfGBq1wqwa80cOMpIQ'
consumer_secret = 'K9HbLB8w4kbGyL7HQPPyuIF4ncSZSRFbgF32li9C30ph1V2jWk'
access_token = '109971381-8jKPS1Pxp4vO7jJRZhBvwDKPtkWWNfnVVYbRGbz6'
access_secret = 'do2rQ4vihB98l8ev1ZbIo8CYWEEI9ePHMpjqB8be4LpE6'

def fetch_tw(ids):
    list_of_tw_status = api.statuses_lookup(ids, tweet_mode= "extended")
    empty_data = pd.DataFrame()
    for status in list_of_tw_status:
            tweet_elem = {"tweet_id": status.id,
                          "screen_name": status.user.screen_name,
                          "tweet": status.full_text,
                          "date": status.created_at,
                          "location": status.user.location,
                          "followers_count": status.user.followers_count,
                          "verified": status.user.verified,
                          "favourites_count": status.user.favourites_count,
                          "retweet_count": status.retweet_count,
                          "language": status.lang
                          }
            empty_data = empty_data.append(tweet_elem, ignore_index = True)
    return empty_data

if __name__ == '__main__':
    # Handle Twitter authetification and the connection to Twitter Streaming API
    list_of_dfs = []
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    path = '../Covid19SentimentStats/Covid19AnnotatedTweetsIds'
    for dir in os.listdir(path):
        print(dir)
        dir_path = os.path.join(path,dir)
        for file in os.listdir(dir_path):
            print(file)
            if os.path.isfile('../Covid19SentimentStats/Tweets/'+dir+'/'+file+'.csv'):
                print('Y')
                pass
            else:
                full_path = os.path.join(dir_path,file)
                ids_df = pd.read_json(full_path, compression='gzip', lines=True)
                ids_df = ids_df[ids_df.location.notnull()]
                ids_df = ids_df[ids_df.apply(lambda x: x['location']['country'] == 'United States', axis=1)]
                ids_list = ids_df['tweet_id'].tolist()
                if len(ids_list) > 5000:
                    ids_list = ids_list[:5000]

                total_count = len(ids_list)
                chunks = (total_count - 1) // 50 + 1
                result_temp = pd.DataFrame()
                for i in range(chunks):
                    batch = ids_list[i*50:(i+1)*50]
                    result = fetch_tw(batch)
                    result_temp = result_temp.append(result)
                result_temp.to_csv('../Covid19SentimentStats/Tweets/'+dir+'/'+file+'.csv')

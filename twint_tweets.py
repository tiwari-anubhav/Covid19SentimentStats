import twint
from datetime import date, timedelta
import os
# twint --since="2020-08-01" --until="2020-08-25" -g="35.1663,-101.8868,3000km" -limit 40000 -pt --hashtags -o file.csv --csv

#os.chdir("../Covid19SentimentStats/2020-08_UK")
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = date(2020,10,1)
end_date = date(2020,10,2)
# Configure
for single_date in daterange(start_date, end_date):
    curr_date = single_date.strftime("%Y-%m-%d")
    next_date = single_date + timedelta(days=1)
    next_date = next_date.strftime("%Y-%m-%d")
    c = twint.Config()
    c.Geo = "51.509865, -0.118092,100km"
    c.Limit = 1
    #c.Store_csv = True
    c.Username = "realDonaldTrump"
    c.Since = curr_date
    c.Until = next_date
    #c.Output = "tweets_" + str(curr_date) + ".csv"
    c.Popular_tweets = True
    c.Lang = "en"
    c.Show_hashtags = True
    #c.Search = "(from:realDonaldTrump)"
    c.Filter_retweets = True
    c.Stats = True
    # Run
    twint.run.Search(c)
    # tlist = c.search_tweet_list
    # print(tlist)



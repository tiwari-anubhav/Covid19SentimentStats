import stanza
import pandas as pd
stanza.download('en')

nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
new_df = pd.read_csv('../Covid19SentimentStats/Data/TestData/TestData_Mask.csv')
new_df['polarity_values'] = None
for index, row in new_df.iterrows():
    sent_dict = {}
    doc = nlp(row['clean_tweet'])
    for i, sentence in enumerate(doc.sentences):
        sent_dict[i] = sentence.sentiment
    lst = list(sent_dict.values())
    sentiment = max(set(lst), key=lst.count)
    new_df.at[index, "polarity_values"] = sentiment
    print(index)
new_df.to_csv('../Covid19SentimentStats/Data/TestData/TestData_Mask.csv', index=False)

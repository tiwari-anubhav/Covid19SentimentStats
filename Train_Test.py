import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def train_test_split_data(data,splitPercent,randomState):

    X = list(data['clean_tweet'])
    y = np.array(list(data['polarity_values']))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitPercent, random_state=randomState)

    return X_train,X_test,y_train,y_test


def tokenize_and_padding(X_train,X_test):
    # Tokenize to  to create a word-to-index dictionary
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return X_train, X_test,vocab_size,tokenizer


def create_embedding_matrix(embeddings_dictionary,vocab_size,tokenizer):
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

def create_word_embeddings(gloveFile):
    embeddings_dictionary = dict()
    glove_file = open(gloveFile, encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()

    return embeddings_dictionary


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    path = 'Data/'
    gloveFile = 'Data/GloVe/glove.twitter.27B.100d.txt'
    train_data_path = 'Data/TrainingData/Tweets_with_mask_All_train.csv'
    val_data_path = 'Data/TrainingData/val.csv'
    test_data_path = 'Data/TrainingData/test.csv'
    data_dir = 'Processed_Tweets/Tweets_with_mask_All.csv'

    data = pd.read_csv(train_data_path)
    # data['polarity_values'] = data.apply(lambda x: TextBlob(x['clean_tweet']).polarity,axis=1)

    X_train, X_test, y_train, y_test = train_test_split_data(data,0.20, 42)
    X_train, X_test, vocab_size, tokenizer = tokenize_and_padding(X_train,X_test)
    embeddings_dict = create_word_embeddings(gloveFile)
    embeddings_mat = create_embedding_matrix(embeddings_dict,vocab_size,tokenizer)

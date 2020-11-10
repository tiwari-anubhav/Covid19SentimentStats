import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
import Train_Test as tt

def create_model_rnn(vocab_size,embedding_matrix,maxlen):
    # create the model
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(10, activation='softmax'))
    # Adam Optimiser
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

def create_sequential_model_rnn(vocab_size,embedding_matrix,maxlen):
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(100, return_sequences=True))

    model.add(LSTM(50))
    model.add(Dense(1, activation='softmax'))

    return model

def train_model(model,train_x, train_y, test_x, test_y, batch_size):
    # save the best model and early stopping
    saveBestModel = keras.callbacks.ModelCheckpoint('../best_weight_glove_bi_100d.hdf5', monitor='acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
    # Fit the model
    model.fit(train_x, train_y, batch_size=batch_size, epochs=25, callbacks=[saveBestModel, earlyStopping])
    # Final evaluation of the model
    score, acc = model.evaluate(test_x, test_y, batch_size=batch_size)
    return model, score, acc


def create_sequential_model(vocab_size,embedding_matrix,maxlen):
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def create_sequential_model_cnn(vocab_size,embedding_matrix,maxlen):
    model = Sequential()

    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))

    return model

if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    path = 'Data/'
    gloveFile = 'Data/GloVe/glove.twitter.27B.100d.txt'
    train_data_path = 'Data/TrainingData/Tweets_with_Social_Distancing_All_train.csv'
    data_dir = 'Processed_Tweets/Tweets_with_mask_All.csv'
    maxlen = 100
    data = pd.read_csv(train_data_path)
    data.clean_tweet=data.clean_tweet.astype(str)
    # data['polarity_values'] = data.apply(lambda x: TextBlob(x['clean_tweet']).polarity,axis=1)

    X_train, X_test, y_train, y_test = tt.train_test_split_data(data,0.20, 42)
    X_train, X_test, vocab_size, tokenizer = tt.tokenize_and_padding(X_train,X_test)
    embeddings_dict = tt.create_word_embeddings(gloveFile)
    embeddings_mat = tt.create_embedding_matrix(embeddings_dict,vocab_size,tokenizer)

    #### Fitting and Creating the model
    # model = create_sequential_model(vocab_size,embeddings_mat,maxlen)
    model = create_sequential_model_cnn(vocab_size,embeddings_mat,maxlen)
    #model = create_sequential_model_rnn(vocab_size,embeddings_mat,maxlen)
    #model = create_model_rnn(vocab_size,embeddings_mat,maxlen)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # history = model.fit(X_train, y_train, batch_size=50, epochs=24, verbose=1, validation_split=0.2)
    model, score, acc = train_model(model,X_train,  y_train, X_test, y_test, 50)
    print(model.summary())
    print(score, acc)
    print("Test Score:", score)
    print("Test Accuracy:", acc)

    ### Prediction of the test data
    # test_data = pd.read_csv(data_dir)
    # tweet_text = test_data.iloc[504]['clean_tweet']
    # print(tweet_text)
    # tweet_text = tokenizer.texts_to_sequences(tweet_text)
    # flat_list = []
    # for sublist in tweet_text:
    #     for item in sublist:
    #         flat_list.append(item)
    #
    # flat_list = [flat_list]

    # instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    # print(model.predict(instance))

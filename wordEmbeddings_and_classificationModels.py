import json
import fasttext
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.layers as layers
from keras.models import Model,load_model
from keras.datasets import imdb
from gensim.models import Word2Vec,FastText
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input,Embedding,Dense,Flatten,Conv2D, Dropout,Concatenate,concatenate,Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, MaxPooling1D,MaxPooling2D,GlobalMaxPooling1D, MaxPool2D, Reshape
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import Sequential
from sklearn.metrics import accuracy_score,classification_report
from tensorflow.keras.layers import Layer
from keras import backend as K
from numpy import array,asarray
from keras.preprocessing.text import one_hot
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
from nltk import pos_tag,word_tokenize


def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set


with open("model_params.json", "r") as f:
    model_params = json.load(f)
model_params = json_to_dict(model_params)

with open("config.json","r") as f:
    params_set = json.load(f)
params_set = json_to_dict(params_set)


def load_imdb_data(vocab_size,max_len):
    """
        Loads the keras imdb dataset

        Args:
            vocab_size = {int} the size of the vocabulary
            max_len = {int} the maximum length of input considered for padding

        Returns:
            X_train = tokenized train data
            X_test = tokenized test data

    """
    INDEX_FROM = 3

    (X_train,y_train),(X_test,y_test) = imdb.load_data(num_words = vocab_size,index_from = INDEX_FROM)

    return X_train,X_test,y_train,y_test


def prepare_data_for_word_vectors_imdb(X_train):
    """
        Prepares the input

        Args:
            X_train = tokenized train data

        Returns:
            sentences = {list} sentences containing words as tokens
            word_index = {dict} word and its indexes in whole of imdb corpus

    """
    INDEX_FROM = 3
    word_to_index = imdb.get_word_index()
    word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}

    word_to_index["<START>"] =1
    word_to_index["<UNK>"]=2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences = []
    for i in range(len(X_train)):
        temp = [index_to_word[ids] for ids in X_train[i]]
        sentences.append(temp)
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_indexes = tokenizer.word_index
    """
    return sentences,word_to_index


def prepare_data_for_word_vectors(X):
    sentences_as_words=[]
    word_to_index={}
    count=1
    for sent in X:
        temp = sent.split()
        sentences_as_words.append(temp)
    for sent in sentences_as_words:
        for word in sent:
            if word_to_index.get(word,None) is None:
                word_to_index[word] = count
                count +=1
    index_to_word = {v:k for k,v in word_to_index.items()}
    sentences=[]
    for i in range(len(sentences_as_words)):
        temp = [word_to_index[w] for w in sentences_as_words[i]]
        sentences.append(temp)


    return sentences_as_words,sentences,word_to_index

def data_prep_ELMo(train_x,train_y,test_x,test_y,max_len,word_ix):

    INDEX_FROM = 0
    #word_to_index = imdb.get_word_index()
    word_to_index = word_ix
    #word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}
    #print(word_to_index)
    word_to_index["<START>"] = 1
    word_to_index["<UNK>"] = 2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences=[]
    for i in range(len(train_x)):
        temp = [index_to_word[ids] for ids in train_x[i]]
        sentences.append(temp)

    test_sentences=[]
    for i in range(len(test_x)):
        temp = [index_to_word[ids] for ids in test_x[i]]
        test_sentences.append(temp)

    train_text = [' '.join(sentences[i][:max_len]) for i in range(len(sentences))]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_y.tolist()

    test_text = [' '.join(test_sentences[i][:500]) for i in range(len(test_sentences))]
    test_text = np.array(test_text , dtype=object)[:, np.newaxis]
    test_label = test_y.tolist()

    return train_text,train_label,test_text,test_label



class Embed:
    def __init__(self,vocab_size,embed_dim,pos_output_dim,max_len,pos_trainable_param):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pos_output_dim=pos_output_dim
        self.pos_input_dim = 20
        self.max_len = max_len
        self.char_to_int = {}
        self.int_to_char ={}
        self.pos_trainable_param = pos_trainable_param


    def embed_sentences(self,word_index,model,trainable_param,X_train_pad):

        embedding_matrix = np.zeros((self.vocab_size,self.embed_dim))
        for word, i in word_index.items():
            try:
                embedding_vector = model[word]
            except:
                pass
            try:
                if embedding_vector is not None:
                    embedding_matrix[i]=embedding_vector
            except:
                pass

        embed_layer = Embedding(self.vocab_size,self.embed_dim,weights =[embedding_matrix],trainable=True)

        input_seq = Input(shape=(X_train_pad.shape[1],))
        embed_seq = embed_layer(input_seq)
        return input_seq,embed_seq

    
    def tag_pos(self,sentences):
        pos_tagged_sent = []
        pos_tagged_sent_all = []
        for sent in sentences:
            pos_tagged_sent.extend(pos_tag(sent))
            pos_tagged_sent_all.append(pos_tag(sent))
        tags = list(set([i[1] for i in pos_tagged_sent]))
        self.pos_input_dim = len(tags)
        self.char_to_int = dict((c, i) for i, c in enumerate(tags))
        self.int_to_char = dict((i, c) for i, c in enumerate(tags))

        X_pos_encoded =[]
        for i in range(len(pos_tagged_sent_all)):
            temp = [self.char_to_int[pos[1]] for pos in pos_tagged_sent_all[i]]
            X_pos_encoded.append(temp)

        return np.array(X_pos_encoded)


    def embed_pos(self,X_pos_arr):
        input_seq_pos = Input(shape=(X_pos_arr.shape[1],))
        embed_seq_pos = Embedding(self.pos_output_dim,self.pos_input_dim,
                                  input_length=self.max_len, 
                                  trainable=self.pos_trainable_param)(input_seq_pos)

        return input_seq_pos,embed_seq_pos
    
    
    def pos_model_build(input_seq,input_seq_pos,embed_seq,embed_seq_pos,
                        pad_train_x,X_pos_arr,train_y,epochs,batch_size,
                        pad_test_x,X_pos_test_arr,test_y):
        
        x = concatenate([embed_seq, embed_seq_pos])
        x = Dense(256,activation ="relu")(x)
        x = Flatten()(x)
        preds = Dense(train_y.shape[1],activation="sigmoid")(x)

        model = Model(inputs=[input_seq, input_seq_pos], outputs=preds)
        loss_param = ""
        if train_y.shape[1] == 2:
            loss_param = model_params["loss"][0]
        elif train_y.shape[1] > 2:
            loss_param = model_params["loss"][1]
        model.compile(loss=loss_param,optimizer="adam",metrics=["accuracy"])

        return model
    

def padding_input(X_train,X_test,maxlen):
    """
        Pads the input upto considered max length

        Args:
            X_train = tokenized train data
            X_test = tokenized test data

        Returns:
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data

    """

    X_train_pad = pad_sequences(X_train,maxlen=maxlen,padding="post")

    X_test_pad = pad_sequences(X_test,maxlen=maxlen,padding="post")

    return X_train_pad,X_test_pad


def building_word_vector_model(option,sentences,embed_dim,workers,window,train_data,X_train,y_train):
    """
        Builds the word vector

        Args:
            type = {int} 0 for Embedding parameter, as the following options:
                0 for Word2vec, 
                1 for gensim Fastext, 
                2 for Fasttext 2018, 
                3 for GloVe, 
                4 for pre-trained Word2vec, 
                5 for word2vec + POS.
                6 for glove+POS
                7 for fasttext+POS
                8 for Elmo+POS
                9 for Elmo only 
            sentences = {list} list of tokenized words
            embed_dim = {int} embedding dimension of the word vectors
            workers = {int} no. of worker threads to train the model (faster training with multicore machines)
            window = {int} max distance between current and predicted word
            y_train = y_train

        Returns:
            model = Word2vec/Gensim fastText/ Fastext_2018 model trained on the training corpus
    """
    # K.clear_session()
    if option in [0,5]:
        print("Training a word2vec model")
        model = Word2Vec(sentences=sentences, workers = workers, window = window) #, size = embed_dim)
        return model

    elif option in [1,7]:
        print("Training a Gensim FastText model")
        model = FastText(sentences=sentences, size = embed_dim, workers = workers, window = window)
        return model
    
    elif option in[2,8]:
        print("Training a Fasttext model from Facebook Research")
        y_train = ["__label__positive" if i==1 else "__label__negative" for i in y_train]

        with open("train_data.txt","w") as text_file:
            for i in range(len(sentences)):
                print(sentences[i],y_train[i],file = text_file)

        model = fasttext.skipgram("train_data.txt","model_fasttext",dim = embed_dim)
        return model
    
    elif option in [3,6]:
        print("Training a GloVe model")
        # prepare tokenizer
        tokenizer = Tokenizer()
        docs = train_data.text
        labels = y_train
        tokenizer.fit_on_texts(docs)
        vocab_size = len(tokenizer.word_index) + 1
        # integer encode the documents
        encoded_docs = tokenizer.texts_to_sequences(docs)
        #print(encoded_docs)
        # pad documents to a max length of 4 words
        max_length = 4
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        #the whole embedding into memory
        embeddings_index  = {}
        f = open('glove.6B.100d.txt',encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            #print(word)
            coefs = asarray(values[1:], dtype='float32')
            #print(coefs)
            embeddings_index[word] = coefs
        f.close()
        
        EMBEDDING_DIM=params_set["embed_dim"]

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, 100))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # define model
        embedding_layer = Embedding(vocab_size,
                                    100,
                                    weights=[embedding_matrix],
                                    trainable=True)

        sequence_length = train_data.shape[1]
        inputs = Input(shape=(sequence_length,))
        embedding = embedding_layer(inputs)
        flatten = Flatten()(embedding)
        dense = Dense(1,activation="sigmoid")(flatten)

        output = Dense(units=y_train.shape[1], activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(dense)

        # this creates a model that includes
        model = Model(inputs, output)
          
        return model
    
    elif option == 4:
        print("Training a pre-trained Word2vec model")
        NUM_WORDS=20000
        tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                              lower=True)
        texts=train_data.text
        tokenizer.fit_on_texts(texts)
        sequences_train = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        word_vectors = KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)

        EMBEDDING_DIM=params_set["embed_dim"]
        
        vocabulary_size=min(len(word_index)+1,NUM_WORDS)
        embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i>=NUM_WORDS:
                continue
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

        del(word_vectors)

        embedding_layer = Embedding(vocabulary_size,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    trainable=True)
        sequence_length = X_train.shape[1]
        filter_sizes = [3,4,5]
        num_filters = model_params["nb_filter"]
        drop = model_params["dropout"]

        inputs = Input(shape=(sequence_length,))
        embedding = embedding_layer(inputs)
        reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

        conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
        conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
        conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

        maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
        maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
        maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

        merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
        flatten = Flatten()(merged_tensor)
        reshape = Reshape((3*num_filters,))(flatten)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=y_train.shape[1], activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

        # this creates a model that includes
        model = Model(inputs, output)
        return model



def cnn_classification_model(x,y,X_train,X_test,y_train,y_test,vocabulary_size):
    """
        Builds the classification model for sentiment analysis

        Args:
            x : data
            y : labels
            X_train  :  train set data
            X_test  :  test set data
            y_train : train set labels
            y_test : test set labels
            vocabulary_size = {int} size of the vocabulary
    """
    sequence_length = X_train.shape[1]
    filter_sizes = [3,4,5]
    num_filters = model_params["nb_filter"]
    drop = model_params["dropout"]
    embedding_dim=params_set["embed_dim"]
    #embedding_matrix = np.zeros((vocabulary_size,embedding_dim))
    epochs = model_params["epochs"] 
    batch_size = model_params["batch_size"]
    

    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), 
                    padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), 
                    padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), 
                    padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=y_train.shape[1], activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    #checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', 
    #                             monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_param = ""
    if y_train.shape[1] == 2:
        loss_param = model_params["loss"][0]
    elif y_train.shape[1] > 2:
        loss_param = model_params["loss"][1]
    model.compile(optimizer=adam,loss=loss_param,metrics= model_params["metrics"] )

    return model      


def lstm_classification_model(embed_dim,X_train_pad,X_test_pad,X_train,y_train,y_test,vocab_size,word_index,w2vmodel,
                         trainable_param,option):
    """
        Builds the classification model for sentiment analysis

        Args:
            embded_dim = {int} dimension of the word vectors
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data
            vocab_size = {int} size of the vocabulary
            word_index =  {dict} word and its indexes in whole of imdb corpus
            w2vmodel = Word2Vec model
            trainable_param = {bool} whether to train the word embeddings in the Embedding layer
            option = {int} choice of word embedding
    """

    embedding_matrix = np.zeros((vocab_size,embed_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = w2vmodel[word]
        except:
            pass
        try:
            if embedding_vector is not None:
                embedding_matrix[i]=embedding_vector
        except:
            pass

    embed_layer = Embedding(vocab_size,embed_dim,weights =[embedding_matrix],trainable=trainable_param)
    input_seq = Input(shape=(X_train_pad.shape[1],))
    embed_seq = embed_layer(input_seq)
    x = Dense(256,activation ="relu")(embed_seq)
    lstm1 = layers.LSTM(model_params["hidden_dims"], return_sequences=True, dropout=model_params["dropout"], recurrent_dropout=model_params["recurrent_dropout"])(x)
    lstm2 = layers.LSTM(model_params["hidden_dims"])(lstm1)
    x = Dense(256,activation ="relu")(lstm2)
    #x = Flatten()(x2)
    preds = Dense(y_train.shape[1],activation="sigmoid")(x)

    model = Model(input_seq,preds)
    loss_param = ""
    if y_train.shape[1] == 2:
        loss_param = model_params["loss"][0]
    elif y_train.shape[1] > 2:
        loss_param = model_params["loss"][1]
    model.compile(loss=loss_param,optimizer=model_params["optimizer"],metrics= model_params["metrics"])

    return model 


def bilstm_classification_model(embed_dim,X_train_pad,X_test_pad,X_train,y_train,y_test,vocab_size,word_index,w2vmodel,
                         trainable_param,option):
    """
        Builds the classification model for sentiment analysis

        Args:
            embded_dim = {int} dimension of the word vectors
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data
            vocab_size = {int} size of the vocabulary
            word_index =  {dict} word and its indexes in whole of imdb corpus
            w2vmodel = Word2Vec model
            trainable_param = {bool} whether to train the word embeddings in the Embedding layer
            option = {int} choice of word embedding
    """
    embedding_matrix = np.zeros((vocab_size,embed_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = w2vmodel[word]
        except:
            pass
        try:
            if embedding_vector is not None:
                embedding_matrix[i]=embedding_vector
        except:
            pass

    embed_layer = Embedding(vocab_size,embed_dim,weights =[embedding_matrix],trainable=trainable_param)
    maxlen = params_set["max_len"]
    

    input_seq = Input(shape=(X_train_pad.shape[1],))
    embed_seq = embed_layer(input_seq)
    print(embed_seq.shape)
    bilstm = layers.Bidirectional(layers.LSTM(model_params["hidden_dims"]))(embed_seq)
    x = Dense(256,activation ="relu")(bilstm)
    preds = Dense(y_train.shape[1],activation="sigmoid")(x)
    model = Model(input_seq,preds)
    loss_param = ""
    if y_train.shape[1] == 2:
        loss_param = model_params["loss"][0]
    elif y_train.shape[1] > 2:
        loss_param = model_params["loss"][1]
    model.compile(loss=loss_param,optimizer=model_params["optimizer"],metrics= model_params["metrics"])
    
    return model  



def classification_model(embed_dim,X_train_pad,X_test_pad,y_train,y_test,vocab_size,word_index,w2vmodel,
                         trainable_param,option):
    """
        Builds the classification model for sentiment analysis

        Args:
            embded_dim = {int} dimension of the word vectors
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data
            vocab_size = {int} size of the vocabulary
            word_index =  {dict} word and its indexes in whole of imdb corpus
            w2vmodel = Word2Vec model
            trainable_param = {bool} whether to train the word embeddings in the Embedding layer
            option = {int} choice of word embedding
    """

    embedding_matrix = np.zeros((vocab_size,embed_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = w2vmodel[word]
        except:
            pass
        try:
            if embedding_vector is not None:
                embedding_matrix[i]=embedding_vector
        except:
            pass

    embed_layer = Embedding(vocab_size,embed_dim,weights =[embedding_matrix],trainable=trainable_param)
    
    input_seq = Input(shape=(X_train_pad.shape[1],))
    embed_seq = embed_layer(input_seq)
    x = Dense(256,activation ="relu")(embed_seq)
    x = Flatten()(x)
    preds = Dense(y_train.shape[1],activation="sigmoid")(x)
    model = Model(input_seq,preds)
    loss_param = ""
    if y_train.shape[1] == 2:
        loss_param = model_params["loss"][0]
    elif y_train.shape[1] > 2:
        loss_param = model_params["loss"][1]    
    print("loss_param: ",loss_param)
    model.compile(loss=loss_param,optimizer=model_params["optimizer"],metrics= model_params["metrics"])
    
    return model

def ELMoEmbedding(x):
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=False
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)
    
    


def Classification_model_with_ELMo(train_text,train_label,test_text,test_label,vocab_size,epochs,batch_size): 
    input_text = layers.Input(shape=(train_text.shape[1],), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
	
    pred = layers.Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_text], outputs=pred)
    
    loss_param = model_params["loss"][1]
	
    # Build Model
    model.compile(loss=loss_param,optimizer=model_params["optimizer"],metrics= model_params["metrics"])
    
    return model 

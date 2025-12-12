import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import *
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Embedding, concatenate
from tensorflow.keras.layers import Dense, Input, Flatten, average, Lambda

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers #keras2
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.optimizers import *


def get_model1(lr,N,NUM_CHANNEL,NUM_CLASS):
    
    image_input = Input(shape=(N,N,NUM_CHANNEL),)

    image_rep = Conv2D(128,(2,2),padding='same')(image_input)
    image_rep = Activation('relu')(image_rep)
    #image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep0 = image_rep
    
    for i in range(1):
        image_rep = Conv2D(128,(2,2),padding='same')(image_rep0)
        image_rep = Activation('relu')(image_rep)
        image_rep = Dropout(0.2)(image_rep)
        image_rep0 = Lambda(lambda x:x[0]+x[1])([image_rep,image_rep0])
        
    image_rep0 = MaxPool2D((2,2))(image_rep0)
    
    for i in range(1):
        image_rep = Conv2D(128,(2,2),padding='same')(image_rep0)
        image_rep = Activation('relu')(image_rep)
        image_rep = Dropout(0.2)(image_rep)
        image_rep0 = Lambda(lambda x:x[0]+x[1])([image_rep,image_rep0])

    image_rep0 = MaxPool2D((2,2))(image_rep0)

    image_rep = Flatten()(image_rep0)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
#     image_rep = Dense(512,activation='relu')(image_rep)
#     image_rep = Dropout(0.2)(image_rep)
#     image_rep = Dense(512,activation='relu')(image_rep)
#     image_rep = Dropout(0.2)(image_rep)
    logit = Dense(NUM_CLASS,activation='softmax')(image_rep)
    
    model = Model(image_input,logit)
    
    model.compile(loss=['categorical_crossentropy'],
                      optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr),
                      metrics=['acc'])

    return model


def get_model2(lr,N,NUM_CHANNEL,NUM_CLASS):
    
    image_input = Input(shape=(28,28,NUM_CHANNEL),)

    image_rep = Conv2D(32,(5,5),)(image_input)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Conv2D(128,(5,5),)(image_rep)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Flatten()(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    logit = Dense(NUM_CLASS,activation='softmax')(image_rep)
    
    model = Model(image_input,logit)
    
    model.compile(loss=['categorical_crossentropy'],
                      optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr),
                      metrics=['acc'])

    return model

def get_model(dataset,lr,N,NUM_CHANNEL,NUM_CLASS):
    if dataset == 'CIFAR10':
        return get_model1(lr,N,NUM_CHANNEL,NUM_CLASS)
    else:
        return get_model2(lr,N,NUM_CHANNEL,NUM_CLASS)


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
import numpy as np

class detect12(object):
        def __init__(self):
                ## build model
                inp = Input(shape=(12,12,3),dtype='float32')
                x = Conv2D(16, (3, 3), strides = 1,padding="same")(inp)
                x = Activation("relu")(x)
                x = MaxPooling2D(pool_size=(3,3),strides=2)(x)
                x = Flatten()(x)
                x = Dense(16)(x)
                x = Activation("relu")(x)
                x = Dense(1)(x)
                x = Activation("sigmoid")(x)
                self.model = Model(inputs=inp,outputs=x)
                self.inp = inp
                self.out = x
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

class detect24(object):
        def __init__(self):
                ## build model 24
                inp24 = Input(shape=(24,24,3),dtype='float32')
                x1 = Conv2D(64, (5, 5), strides = 1,padding="same")(inp24)
                x1 = Activation("relu")(x1)
                x1 = MaxPooling2D(pool_size=(3,3),strides=2)(x1)
                x1 = Flatten()(x1)
                x1 = Dense(128)(x1)


                inp12 = Input(shape=(12,12,3),dtype='float32')
                x2 = Conv2D(16, (3, 3), strides = 1,padding="same")(inp12)
                x2 = Activation("relu")(x2)
                x2 = MaxPooling2D(pool_size=(3,3),strides=2)(x2)
                x2 = Flatten()(x2)
                x2 = Dense(16)(x2)

                x = concatenate([x1,x2])
                x = Activation("relu")(x)
                x = Dense(8)(x)
                x = Activation("relu")(x)
                x = Dense(1)(x)
                x = Activation("sigmoid")(x)

                self.model = Model(inputs=[inp24,inp12],outputs=x)
                self.inp = [inp24,inp12]
                self.out = x
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer=Adam(
                                       lr =0.0001 
                          ),
                          metrics=['accuracy'])
class detect48(object):
        def __init__(self):
                ## build model 48
                inp48 = Input(shape=(48,48,3),dtype='float32')
                x1 = Conv2D(64, (5, 5), strides = 1,padding="same")(inp48)
                x1 = Activation("relu")(x1)
                x1 = MaxPooling2D(pool_size=(3,3),strides=2)(x1)
                #x1 = BatchNormalization()(x1)
                x1 = Conv2D(64,(5,5),strides = 1,padding = "same")(x1)
                x1 = Activation("relu")(x1)
                #x1 = BatchNormalization()(x1)
                x1 = MaxPooling2D(pool_size=(3,3),strides=2)(x1)
                x1 = Flatten()(x1)
                x1 = Dense(256)(x1)

                inp24 = Input(shape=(24,24,3),dtype='float32')
                x2 = Conv2D(64, (5, 5), strides = 1,padding="same")(inp24)
                x2 = Activation("relu")(x2)
                x2 = MaxPooling2D(pool_size=(3,3),strides=2)(x2)
                x2 = Flatten()(x2)
                x2 = Dense(128)(x2)

                inp12 = Input(shape=(12,12,3),dtype='float32')
                x3 = Conv2D(16, (3, 3), strides = 1,padding="same")(inp12)
                x3 = Activation("relu")(x3)
                x3 = MaxPooling2D(pool_size=(3,3),strides=2)(x3)
                x3 = Flatten()(x3)
                x3 = Dense(16)(x3)

                x = concatenate([x1,x2,x3])
                x = Activation("relu")(x)
                x = Dense(128)(x)
                x = Activation("relu")(x)
                x = Dense(1)(x)
                x = Activation("sigmoid")(x)
                self.model = Model(inputs=[inp48, inp24, inp12],outputs=x)
                self.inp = [inp48, inp24, inp12]
                self.out = x
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer=SGD(
                                       lr =0.01 
                          ),
                          metrics=['accuracy'])
class calib12(object):
        def __init__(self):
                ## build model
                inp = Input(shape=(12,12,3),dtype='float32')
                x = Conv2D(16, (3, 3), strides = 1,padding="same")(inp)
                x = Activation("relu")(x)
                x = MaxPooling2D(pool_size=(3,3),strides=2)(x)
                x = Flatten()(x)
                x = Dense(128)(x)
                x = Activation("relu")(x)
                x = Dense(45)(x)
                x = Activation("softsign")(x)
                self.model = Model(inputs=inp,outputs=x)
                self.inp = inp
                self.out = x
        def loss(self,y_true,y_pred):
                K.print_tensor(y_true,message="y_true is")
                return (y_true-y_pred)**2
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[categorical_accuracy])

class calib24(object):
        def __init__(self):
                ## build model
                inp = Input(shape=(24,24,3),dtype='float32')
                x = Conv2D(32, (5, 5), strides = 1,padding="same")(inp)
                x = Activation("relu")(x)
                x = MaxPooling2D(pool_size=(3,3),strides=2)(x)
                x = Flatten()(x)
                x = Dense(64)(x)
                x = Activation("relu")(x)
                x = Dense(45)(x)
                x = Activation("softsign")(x)
                self.model = Model(inputs=inp,outputs=x)
                self.inp = inp
                self.out = x
        def loss(self,y_true,y_pred):
                K.print_tensor(y_true,message="y_true is")
                return (y_true-y_pred)**2
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer=Adam(
                                       lr =0.0001 
                          ),
                          metrics=[categorical_accuracy])
class calib48(object):
        def __init__(self):
                ## build model
                inp = Input(shape=(48,48,3),dtype='float32')
                x = Conv2D(64, (5, 5), strides = 1,padding="same")(inp)
                x = Activation("relu")(x)
                x = MaxPooling2D(pool_size=(3,3),strides=2)(x)
                x = BatchNormalization()(x)
                x = Conv2D(64, (5, 5), strides = 1,padding="same")(inp)
                x = Activation("relu")(x)
                x = BatchNormalization()(x)
                x = Flatten()(x)
                x = Dense(256)(x)
                x = Activation("relu")(x)
                x = Dense(45)(x)
                x = Activation("softsign")(x)
                self.model = Model(inputs=inp,outputs=x)
                self.inp = inp
                self.out = x
        def loss(self,y_true,y_pred):
                K.print_tensor(y_true,message="y_true is")
                return (y_true-y_pred)**2
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer=SGD(
                                       lr =0.0001 
                          ),
                          metrics=[categorical_accuracy])
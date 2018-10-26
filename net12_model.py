from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam, SGD
from keras import backend as K
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
                          optimizer=Adam(
                                       lr =0.0001 
                          ),
                          metrics=[categorical_accuracy])

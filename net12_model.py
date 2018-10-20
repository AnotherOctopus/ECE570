from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

class detect12(object):
        def __init__(self):
                # dimensions of our images.
                input_shape = 12,12,3
                ## build model
                model = Sequential()
                model.add(Conv2D(16, (3, 3), strides = 1,input_shape=input_shape))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

                model.add(Flatten())
                model.add(Dense(128))
                model.add(Activation("relu"))
                model.add(Dense(45))
                model.add(Activation("relu"))
                self.model = model
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

class calib12(object):
        def __init__(self):
                # dimensions of our images.
                input_shape = 12,12,3
                ## build model
                model = Sequential()
                model.add(Conv2D(16, (3, 3), strides = 1,input_shape=input_shape))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

                model.add(Flatten())
                model.add(Dense(128))
                model.add(Activation("relu"))
                model.add(Dense(45))
                model.add(Activation("relu"))
                self.model = model
        def compile(self):

                self.model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

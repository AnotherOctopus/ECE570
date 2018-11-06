import os
from keras.layers import  concatenate
from keras.layers import Input, Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras.models import Model,load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from config import *
from datagenerators import *
from scipy.misc import imread, imsave
import numpy as np
import os
from utils import *

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
    def train(self,saveas,train_data_dir,valid_data_dir,tags):

        # dimensions of images
        img_width, img_height = 24,24
        # data
        nb_train_data_dir = len(os.listdir(os.path.join(train_data_dir,tags[0]))) + len(os.listdir(os.path.join(train_data_dir,tags[1])))
        nb_valid_data_dir = len(os.listdir(os.path.join(valid_data_dir,tags[0]))) + len(os.listdir(os.path.join(valid_data_dir,tags[1])))
        n_epochs = 50
        batch_size = 128

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)




        # data augmentation
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                        horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator=generate_generator_two(generator=train_datagen,
                                            datadir=train_data_dir,
                                            batch_size=batch_size,
                                            tag=tags)
        valid_generator=generate_generator_two(generator=valid_datagen,
                                            datadir=valid_data_dir,
                                            batch_size=batch_size,
                                            tag=tags)
        """
        train24 /= 255.0
        train12 /= 255.0
        valid24 /= 255.0
        valid12 /= 255.0
        """
        # Train
        hist = self.model.fit_generator(train_generator,
                            steps_per_epoch= nb_train_data_dir// batch_size,
                            epochs=n_epochs,
                            validation_data=valid_generator,
                            validation_steps=nb_valid_data_dir// batch_size,
                            verbose=1)

        self.model.save(saveas)
        return hist
    def test(self,testfile):
            model = load_model('detect24.h5')
            rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
            wind12 = resizetoshape(rawimg,(L1SIZE,L1SIZE))
            rawimg = rawimg[np.newaxis,...]
            predictions =  model.predict([rawimg,wind12])
            print predictions

            model = load_model('net12.h5')
if __name__ == "__main__":
    d24 = detect24()
    d24.test("/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/detect24/train/notface/2.jpg")
    #FACE IS 1

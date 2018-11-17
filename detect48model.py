from keras.layers import  concatenate
from keras.layers import Input, Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras.models import Model, load_model
from keras import backend as K
from utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from datagenerators import *
import os
from scipy.misc import imread, imsave
import numpy as np
import os
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
    def train(self,saveas,train_data_dir,valid_data_dir,tags=["face","notface"]):

        # dimensions of images
        img_width, img_height = 48,48
        # data
        nb_train_data_dir = len(os.listdir(os.path.join(train_data_dir,tags[0]))) + len(os.listdir(os.path.join(train_data_dir,tags[1])))
        nb_valid_dir = len(os.listdir(os.path.join(valid_data_dir,tags[0]))) + len(os.listdir(os.path.join(valid_data_dir,tags[1])))
        n_epochs = 5
        batch_size = 128

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)


        ## build model
        d48 = detect48()
        d48.compile()
        model = d48.model


        # data augmentation
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                        horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)

        # data augmentation
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                        horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator=generate_generator_three(generator=train_datagen,
                                            datadir=train_data_dir,
                                            batch_size=batch_size,
                                            tag=tags
                                            )
        valid_generator=generate_generator_three(generator=valid_datagen,
                                            datadir=valid_data_dir,
                                            batch_size=batch_size,
                                            tag=tags
                                            )
        """
        train24 /= 255.0
        train12 /= 255.0
        valid24 /= 255.0
        valid12 /= 255.0
        """
        # Train
        hist = model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_data_dir // batch_size,
                            epochs=n_epochs,
                            validation_data=valid_generator,
                            validation_steps=nb_valid_dir // batch_size,
                            verbose=1)

        model.save(saveas)
        return hist
    def test(self,testfile):
            model = load_model('detect48.h5')
            rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
            wind24 = resizetoshape(rawimg,(L2SIZE,L2SIZE))
            wind12 = resizetoshape(rawimg,(L1SIZE,L1SIZE))
            rawimg = rawimg[np.newaxis,...]
            predictions =  model.predict([rawimg,wind24,wind12])
            print predictions

if __name__ == "__main__":
    d48 = detect48()
    d48.test("/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/detect48/train/notface/5.jpg")
    #FACE IS 1

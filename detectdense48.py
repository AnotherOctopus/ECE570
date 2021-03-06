from keras.layers import  concatenate, BatchNormalization,AveragePooling2D
from keras.layers import Input, Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator 
from PIL import Image
from keras.optimizers import SGD,Adam
from datagenerators import *
from scipy.misc import imread, imsave
import numpy as np
import os
class dense48(object):
    def __init__(self,growth = 4):
            ## build model 48
            feat = 3
            inp48 = Input(shape=(48,48,feat),dtype='float32')
            inp24 = Input(shape=(24,24,feat),dtype='float32')
            inp12 = Input(shape=(12,12,feat),dtype='float32')
            self.growth = growth
            x1 = Conv2D(growth*12,(5,5),padding="same")(inp48)
            x1 = Activation("relu")(inp48)

            x1 = self.denselayer(x1,feat)
            feat += growth
            x2 = self.denselayer(x1,feat)
            feat += growth
            x3 = concatenate([x1,x2])

            x4 = self.denselayer(x3, feat)
            feat += growth 
            x5 = concatenate([x1,x2,x4])

            x6 = self.denselayer(x5, feat)
            feat += growth

            feat /= 2
            x11 = self.transitionlayer(x6,feat)
            x11 = concatenate([x11,inp24])

            x12 = self.denselayer(x11,feat)
            feat += growth

            x13 = self.denselayer(x12,feat)
            feat += growth

            x14 = concatenate([x12,x13])

            x15 = self.denselayer(x14,feat)
            feat += growth

            x16 = concatenate([x12,x13,x15])

            x17 = self.denselayer(x16,feat)
            feat += growth

            x18 = concatenate([x12,x13,x15,x17])

            x19 = self.denselayer(x18,feat)
            feat += growth

            x20 = concatenate([x12,x13,x15,x17,x19])

            x21 = self.denselayer(x20,feat)
            feat += growth

            x22 = concatenate([x12,x13,x15,x17,x19,x21])

            x23 = self.denselayer(x22,feat)
            feat += growth

            x24 = concatenate([x12,x13,x15,x17,x19,x21,x23])

            x25 = self.denselayer(x24,feat)
            feat += growth

            feat /= 2
            x34 = self.transitionlayer(x25,feat)
            x34 = concatenate([x34,inp12])

            x35 = self.denselayer(x34,feat)
            feat += growth

            x36 = self.denselayer(x35,feat)
            feat += growth

            x37 = concatenate([x35,x36])

            x38 = self.denselayer(x37,feat)
            feat += growth

            x39 = concatenate([x35,x36,x38])

            x40 = self.denselayer(x39,feat)
            feat += growth

            x41 = concatenate([x35,x36,x38,x40])

            x42 = self.denselayer(x41,feat)
            feat += growth

            x43 = concatenate([x35,x36,x38,x40,x42])

            x44 = self.denselayer(x43,feat)
            feat += growth

            x45 = concatenate([x35,x36,x38,x40,x42,x44])

            x46 = self.denselayer(x45,feat)
            feat += growth

            x47 = concatenate([x35,x36,x38,x40,x42,x44,x46])

            x48 = self.denselayer(x47,feat)
            feat += growth

            x49 = concatenate([x35,x36,x38,x40,x42,x44,x46,x48])

            x50 = self.denselayer(x49,feat)
            feat += growth

            x51 = concatenate([x35,x36,x38,x40,x42,x44,x46,x48,x50])

            x52 = self.denselayer(x51,feat)
            feat += growth

            x53 = concatenate([x35,x36,x38,x40,x42,x44,x46,x48,x50,x52])

            x54 = self.denselayer(x53,feat)
            feat += growth

            x55 = concatenate([x35,x36,x38,x40,x42,x44,x46,x48,x50,x52,x54])

            x56 = self.denselayer(x55,feat)
            feat += growth

            feat /= 2
            x57 = self.transitionlayer(x56,feat)

            x = GlobalAveragePooling2D()(x57)
            x = Dense(1000)(x)
            x = Activation("relu")(x)
            x = Dense(100)(x)
            x = Activation("relu")(x)
            x = Dense(1)(x)

            self.model = Model(inputs=[inp48,inp24,inp12],outputs=x)
            self.inp = inp48
            self.out = x34
    def denselayer(self,x,numfeat):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(numfeat,(3,3),padding="same")(x)
        return x
        
    def transitionlayer(self,x,numfeat):
        x = BatchNormalization()(x)
        x = Conv2D(numfeat,(1,1))(x)
        x = AveragePooling2D(pool_size=(2, 2),padding="same")(x)
        return x
    def compile(self):


            self.model.compile(loss='binary_crossentropy',
                        optimizer=Adam(lr=0.0001),
                        metrics=['accuracy'])
    def train(self):

        # dimensions of images
        img_width, img_height = 48,48
        # data
        train_data_dir = "data/faces/detect48/train"
        valid_data_dir = "data/faces/detect48/validation"
        nb_train_samples = len(os.listdir(train_data_dir +"/face")) + len(os.listdir(train_data_dir +"/notface"))
        nb_validation_samples = len(os.listdir(valid_data_dir +"/face")) + len(os.listdir(valid_data_dir +"/notface"))
        n_epochs = 40
        batch_size = 64

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)



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
                                            batch_size=batch_size)
        valid_generator=generate_generator_three(generator=valid_datagen,
                                            datadir=valid_data_dir,
                                            batch_size=batch_size)
        """
        train24 /= 255.0
        train12 /= 255.0
        valid24 /= 255.0
        valid12 /= 255.0
        """
        # Train
        hist = self.model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=n_epochs,
                            validation_data=valid_generator,
                            validation_steps=nb_validation_samples // batch_size,
                            verbose=1)

        self.model.save("densedetect48.h5")
        return hist
    def test(self,testfile):
            model = load_model('detect48.h5')
            rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
            rawimg = rawimg[np.newaxis,...]

            wind24 = Image.fromarray(np.uint8(imread(testfile,mode='RGB')*255)).resize((L2SIZE,L2SIZE))
            wind24 = np.asarray(wind24).astype(np.float32)/255
            wind24 = np.reshape(wind24,(1,L2SIZE,L2SIZE,3))

            wind12 = Image.fromarray(np.uint8(imread(testfile,mode='RGB')*255)).resize((L1SIZE,L1SIZE))
            wind12 = np.asarray(wind12).astype(np.float32)/255
            wind12 = np.reshape(wind12,(1,L1SIZE,L1SIZE,3))

            predictions =  model.predict([rawimg, wind24, wind12])

            print predictions
if __name__ == "__main__":
    d48 = detect48()
    d48.test("/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/detect48/train/notface/2.jpg")
    #FACE IS 1

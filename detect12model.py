from keras.layers import Input, Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from utils import  *
from scipy.misc import imread, imsave
import numpy as np
import os
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
    def train(self,saveas,train_data_dir,validation_data_dir,tags=["face","notface"]):

        # dimensions of images
        img_width, img_height = 12,12
        # data
        nb_train_samples = len(os.listdir(os.path.join(train_data_dir,tags[0]))) + len(os.listdir(os.path.join(train_data_dir,tags[1])))
        nb_validation_samples = len(os.listdir(os.path.join(validation_data_dir,tags[0]))) + len(os.listdir(os.path.join(validation_data_dir,tags[1])))
        n_epochs = 50
        batch_size = 128

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        # data augmentation
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                        horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                                target_size=(img_width, img_height),
                                                                batch_size=batch_size,
                                                                class_mode='binary')

        print self.model.summary()
        # Train
        hist = self.model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=n_epochs,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples // batch_size,
                            shuffle=True,
                            verbose=1)

        self.model.save(saveas)
        
        return hist
    def test(self,testfile):
            model = load_model('net12.h5')
            rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
            rawimg = resizetoshape(rawimg,(L1SIZE,L1SIZE))
            predictions =  model.predict(rawimg)


            print predictions
if __name__ == "__main__":
    d12 = detect12()
    d12.test("/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/detect12/train/notface/3.jpg")
    # face is 0

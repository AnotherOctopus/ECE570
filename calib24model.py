from keras.layers import Input, Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD
import os
from keras.models import Model
from keras.optimizers import Adam
from utils import * 
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
                        optimizer=SGD(),
                        metrics=[categorical_accuracy])
    def train(self,saveas,train_data_dir,validation_data_dir,tags=["face","notface"]):

        # dimensions of images
        img_width, img_height = 24,24
        nb_train_samples = len(os.listdir(train_data_dir +"/face"))
        nb_validation_samples = len(os.listdir(validation_data_dir +"/face"))
        n_epochs = 40
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        ## build model
        c24 = calib24()
        c24.compile()
        model = c24.model

        # data augmentation
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        trainimgs, trainlabels = collecttagsfromdir( train_data_dir, tag = tags[0],shape = 24)
        trainlabels = to_categorical(trainlabels)
        train_generator = train_datagen.flow(trainimgs,
                                            trainlabels)

        validimgs, validlabels = collecttagsfromdir(validation_data_dir, tag = tags[0],shape=24)
        validlabels = to_categorical(validlabels)
        validation_generator = test_datagen.flow(validimgs,
                                                validlabels)
        # Train
        hist = model.fit(trainimgs,
                trainlabels,
                batch_size =64,
                epochs=n_epochs,
                validation_data=(validimgs,validlabels),
                shuffle=True,
                verbose=1)

        model.save("calib24.h5")
        return hist
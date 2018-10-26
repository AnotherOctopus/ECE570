import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import np_utils,to_categorical
from scipy.misc import imread, imsave, imresize
from sklearn.preprocessing import LabelEncoder
from net12_model import detect12,calib12
from config import *
import numpy as np

def trainadj12():
    # dimensions of images
    img_width, img_height = 12,12
    # data
    train_data_dir = "data/adj12/train"
    validation_data_dir = "data/adj12/validation"
    nb_train_samples = len(os.listdir(train_data_dir +"/face"))
    nb_validation_samples = len(os.listdir(validation_data_dir +"/face"))
    n_epochs = 1000
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    ## build model
    c12 = calib12()
    c12.compile()
    model = c12.model

    # data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    trainimgs, trainlabels = collecttagsfromdir( train_data_dir)
    trainlabels = to_categorical(trainlabels)
    train_generator = train_datagen.flow(trainimgs,
                                         trainlabels)

    validimgs, validlabels = collecttagsfromdir(validation_data_dir)
    validlabels = to_categorical(validlabels)
    validation_generator = test_datagen.flow(validimgs,
                                             validlabels)
    trainimgs /= 255.0
    validimgs /= 255.0
    # Train
    model.fit(trainimgs,
              trainlabels,
              batch_size =64,
              epochs=n_epochs,
              validation_data=(validimgs,validlabels),
              shuffle=True,
              verbose=1)

    model.save("calib12.h5")

def collecttagsfromdir(datadir):
    numsamples = len(os.listdir(datadir +"/face"))
    imgs = np.empty((numsamples,12,12,3),dtype=np.float32)
    labels = []
    for idx, img, tag in zip(range(numsamples), os.listdir(datadir+"/face"),os.listdir(datadir+"/tag")):
        imgs[idx,:,:,:] = imread(os.path.join(datadir,"face",img))
        with open(os.path.join(datadir,"tag",tag),"r") as fh:
            labels.append(adjclass.index(fh.read()))
    return imgs, labels

def trainnet12():

    # dimensions of images
    img_width, img_height = 12,12
    # data
    train_data_dir = "data/detect12/train"
    validation_data_dir = "data/detect12/validation"
    nb_train_samples = len(os.listdir(train_data_dir +"/face")) + len(os.listdir(train_data_dir +"/notface"))
    nb_validation_samples = len(os.listdir(validation_data_dir +"/face")) + len(os.listdir(validation_data_dir +"/notface"))
    print nb_train_samples
    n_epochs = 30
    batch_size = 500

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    ## build model
    d12 = detect12()
    d12.compile()
    model = d12.model


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

    print model.summary()
    # Train
    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        shuffle=True,
                        verbose=1)

    model.save("net12.h5")

if __name__ == "__main__":
    trainnet12()
    #trainadj12()

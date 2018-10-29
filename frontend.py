import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import np_utils,to_categorical
from scipy.misc import imread, imsave, imresize
from sklearn.preprocessing import LabelEncoder
from net12_model import detect12,calib12,detect24,calib24,detect48,calib48
from config import *
import numpy as np
from random import shuffle
from PIL import Image
from datagenerators import *
def trainadj48():
    # dimensions of images
    img_width, img_height = 48,48
    # data
    train_data_dir = "data/adj48/train"
    validation_data_dir = "data/adj48/validation"
    nb_train_samples = len(os.listdir(train_data_dir +"/face"))
    nb_validation_samples = len(os.listdir(validation_data_dir +"/face"))
    n_epochs = 100
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    ## build model
    c48 = calib48()
    c48.compile()
    model = c48.model

    # data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    trainimgs, trainlabels = collecttagsfromdir( train_data_dir,shape = 48)
    trainlabels = to_categorical(trainlabels)
    train_generator = train_datagen.flow(trainimgs,
                                         trainlabels)

    validimgs, validlabels = collecttagsfromdir(validation_data_dir,shape=48)
    validlabels = to_categorical(validlabels)
    validation_generator = test_datagen.flow(validimgs,
                                             validlabels)
    # Train
    model.fit(trainimgs,
              trainlabels,
              batch_size =128,
              epochs=n_epochs,
              validation_data=(validimgs,validlabels),
              shuffle=True,
              verbose=1)

    model.save("calib48.h5")
def trainadj24():
    # dimensions of images
    img_width, img_height = 24,24
    # data
    train_data_dir = "data/adj24/train"
    validation_data_dir = "data/adj24/validation"
    nb_train_samples = len(os.listdir(train_data_dir +"/face"))
    nb_validation_samples = len(os.listdir(validation_data_dir +"/face"))
    n_epochs = 1000
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

    trainimgs, trainlabels = collecttagsfromdir( train_data_dir,shape = 24)
    trainlabels = to_categorical(trainlabels)
    train_generator = train_datagen.flow(trainimgs,
                                         trainlabels)

    validimgs, validlabels = collecttagsfromdir(validation_data_dir,shape=24)
    validlabels = to_categorical(validlabels)
    validation_generator = test_datagen.flow(validimgs,
                                             validlabels)
    # Train
    model.fit(trainimgs,
              trainlabels,
              batch_size =64,
              epochs=n_epochs,
              validation_data=(validimgs,validlabels),
              shuffle=True,
              verbose=1)

    model.save("calib24.h5")
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
    # Train
    model.fit(trainimgs,
              trainlabels,
              batch_size =64,
              epochs=n_epochs,
              validation_data=(validimgs,validlabels),
              shuffle=True,
              verbose=1)

    model.save("calib12.h5")

def collecttagsfromdir(datadir,shape = 12):
    numsamples = len(os.listdir(datadir +"/face"))
    imgs = np.empty((numsamples,shape,shape,3),dtype=np.float32)
    labels = []
    for idx, img, tag in zip(range(numsamples), os.listdir(datadir+"/face"),os.listdir(datadir+"/tag")):
        imgs[idx,:,:,:] = imread(os.path.join(datadir,"face",img))
        with open(os.path.join(datadir,"tag",tag),"r") as fh:
            labels.append(adjclass.index(fh.read()))
    return imgs, labels

def trainnet48():

    # dimensions of images
    img_width, img_height = 48,48
    # data
    train_data_dir = "data/detect48/train"
    valid_data_dir = "data/detect48/validation"
    nb_train_samples = len(os.listdir(train_data_dir +"/face")) + len(os.listdir(train_data_dir +"/notface"))
    nb_validation_samples = len(os.listdir(valid_data_dir +"/face")) + len(os.listdir(valid_data_dir +"/notface"))
    n_epochs = 20
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
    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=n_epochs,
                        validation_data=valid_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        verbose=1)

    model.save("detect48.h5")
def trainnet24():

    # dimensions of images
    img_width, img_height = 24,24
    # data
    train_data_dir = "data/detect24/train"
    valid_data_dir = "data/detect24/validation"
    nb_train_samples = len(os.listdir(train_data_dir +"/face")) + len(os.listdir(train_data_dir +"/notface"))
    nb_validation_samples = len(os.listdir(valid_data_dir +"/face")) + len(os.listdir(valid_data_dir +"/notface"))
    n_epochs = 40
    batch_size = 128

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    ## build model
    d24 = detect24()
    d24.compile()
    model = d24.model


    # data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator=generate_generator_two(generator=train_datagen,
                                          datadir=train_data_dir,
                                          batch_size=batch_size)
    valid_generator=generate_generator_two(generator=valid_datagen,
                                          datadir=valid_data_dir,
                                          batch_size=batch_size)
    """
    train24 /= 255.0
    train12 /= 255.0
    valid24 /= 255.0
    valid12 /= 255.0
    """
    # Train
    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=n_epochs,
                        validation_data=valid_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        verbose=1)

    model.save("detect24.h5")

def trainnet12():

    # dimensions of images
    img_width, img_height = 12,12
    # data
    train_data_dir = "data/detect12/train"
    validation_data_dir = "data/detect12/validation"
    nb_train_samples = len(os.listdir(train_data_dir +"/face")) + len(os.listdir(train_data_dir +"/notface"))
    nb_validation_samples = len(os.listdir(validation_data_dir +"/face")) + len(os.listdir(validation_data_dir +"/notface"))
    n_epochs = 20
    batch_size = 128

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
    #trainnet12()
    #trainnet24()
    #trainnet48()
    #trainadj12()
    #trainadj24()
    trainadj48()

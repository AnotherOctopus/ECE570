import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import np_utils,to_categorical
from scipy.misc import imread, imsave, imresize
from sklearn.preprocessing import LabelEncoder
from config import *
import numpy as np
from random import shuffle
from datagenerators import *
def prepnet24data(datadir):
    numsamples = len(os.listdir(datadir +"/face")) + len(os.listdir(datadir +"/notface"))
    imgs24 = np.empty((numsamples,24,24,3),dtype=np.float32)
    imgs12 = np.empty((numsamples,12,12,3),dtype=np.float32)
    labels = np.empty((numsamples,1),dtype=np.float32)
    alldatafiles = [os.path.join(datadir,"face",f) for f in os.listdir(datadir+"/face")] + [os.path.join(datadir,"notface",f) for f in os.listdir(datadir+"/notface")]
    shuffle(alldatafiles)
    for idx, img in enumerate(alldatafiles):
        imgs24[idx,:,:,:] = imread(img)
        imgs12[idx,:,:,:] = np.asarray(Image.open(img).resize((12,12)))
        if "notface" in img:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return imgs24, imgs12, labels
def prepnet48data(datadir):
    numsamples = len(os.listdir(datadir +"/face")) + len(os.listdir(datadir +"/notface"))
    imgs48 = np.empty((numsamples,48,48,3),dtype=np.float32)
    imgs24 = np.empty((numsamples,24,24,3),dtype=np.float32)
    imgs12 = np.empty((numsamples,12,12,3),dtype=np.float32)
    labels = np.empty((numsamples,1),dtype=np.float32)
    alldatafiles = [os.path.join(datadir,"face",f) for f in os.listdir(datadir+"/face")] + [os.path.join(datadir,"notface",f) for f in os.listdir(datadir+"/notface")]
    shuffle(alldatafiles)
    for idx, img in enumerate(alldatafiles):
        imgs48[idx,:,:,:] = imread(img)
        imgs24[idx,:,:,:] = np.asarray(Image.open(img).resize((24,24)))
        imgs12[idx,:,:,:] = np.asarray(Image.open(img).resize((12,12)))
        if "notface" in img:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return imgs48, imgs24, imgs12, labels

def generate_generator_two(generator,datadir, batch_size):
    img24, img12, labels = prepnet24data(datadir)
    genX1 = generator.flow(img24,
                           labels,
                           batch_size = batch_size
                           )
    genX2 = generator.flow(img12,
                           labels,
                           batch_size = batch_size
                           )
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label
            
def generate_generator_three(generator,datadir, batch_size):
    img48, img24, img12, labels = prepnet48data(datadir)
    genX1 = generator.flow(img48,
                           labels,
                           batch_size = batch_size
                           )
    genX2 = generator.flow(img24,
                           labels,
                           batch_size = batch_size
                           )
    genX3 = generator.flow(img12,
                           labels,
                           batch_size = batch_size
                           )
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            yield [X1i[0], X2i[0],X3i[0]], X2i[1]  #Yield both images and their mutual label
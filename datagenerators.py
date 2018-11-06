import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import np_utils,to_categorical
from scipy.misc import imread, imsave, imresize
from sklearn.preprocessing import LabelEncoder
from skimage.transform import pyramid_reduce
from config import *
import numpy as np
from random import shuffle
from datagenerators import *
def prepnet24data(datadir,tag= ["face","notface"]):
    numsamples = len(os.listdir(os.path.join(datadir,tag[0]))) + len(os.listdir(os.path.join(datadir, tag[1])))
    imgs24 = np.empty((numsamples,24,24,3),dtype=np.float32)
    imgs12 = np.empty((numsamples,12,12,3),dtype=np.float32)
    labels = np.empty((numsamples,1),dtype=np.float32)
    alldatafiles = [os.path.join(datadir,tag[0],f) for f in os.listdir(os.path.join(datadir,tag[0]))] + [os.path.join(datadir,tag[1],f) for f in os.listdir(os.path.join(datadir,tag[1]))]
    shuffle(alldatafiles)
    for idx, img in enumerate(alldatafiles):
        frame = imread(img)
        imgs24[idx,:,:,:] = frame
        imgs12[idx,:,:,:] = np.asarray(pyramid_reduce(frame,downscale=2))
        if tag[1] in img:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return imgs24, imgs12, labels
def prepnet48data(datadir, tag= ["face","notface"]):
    numsamples = len(os.listdir(os.path.join(datadir,tag[0]))) + len(os.listdir(os.path.join(datadir, tag[1])))
    imgs48 = np.empty((numsamples,48,48,3),dtype=np.float32)
    imgs24 = np.empty((numsamples,24,24,3),dtype=np.float32)
    imgs12 = np.empty((numsamples,12,12,3),dtype=np.float32)
    labels = np.empty((numsamples,1),dtype=np.float32)
    alldatafiles = [os.path.join(datadir,tag[0],f) for f in os.listdir(os.path.join(datadir,tag[0]))] + [os.path.join(datadir,tag[1],f) for f in os.listdir(os.path.join(datadir,tag[1]))]
    shuffle(alldatafiles)
    for idx, img in enumerate(alldatafiles):
        frame = imread(img)
        imgs48[idx,:,:,:] = frame
        imgs24[idx,:,:,:] = np.asarray(np.asarray(pyramid_reduce(frame,downscale=2)))
        imgs12[idx,:,:,:] = np.asarray(np.asarray(pyramid_reduce(frame,downscale=4)))
        if tag[1] in img:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return imgs48, imgs24, imgs12, labels

def generate_generator_two(generator,datadir, batch_size,tag):
    img24, img12, labels = prepnet24data(datadir,tag)
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
            
def generate_generator_three(generator,datadir, batch_size,tag):
    img48, img24, img12, labels = prepnet48data(datadir,tag)
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

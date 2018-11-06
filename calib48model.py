from keras.layers import Input, Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD
from utils import * 
from keras import backend as K
import os

class calib48(object):
    def __init__(self):
            ## build model
            inp = Input(shape=(48,48,3),dtype='float32')
            x = Conv2D(64, (5, 5), strides = 1,padding="same")(inp)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size=(3,3),strides=2)(x)
            x = BatchNormalization()(x)
            x = Conv2D(64, (5, 5), strides = 1,padding="same")(inp)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(256)(x)
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
        img_width, img_height = 48,48
        nb_train_samples = len(os.listdir(os.path.join(train_data_dir,tags[0])))
        nb_validation_samples = len(os.listdir(os.path.join(validation_data_dir,tags[0])))
        n_epochs = 40
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

        trainimgs, trainlabels = collecttagsfromdir( train_data_dir, tag = tags[0],shape = 48)
        trainlabels = to_categorical(trainlabels)
        train_generator = train_datagen.flow(trainimgs,
                                            trainlabels)

        validimgs, validlabels = collecttagsfromdir(validation_data_dir, tag = tags[0],shape=48)
        validlabels = to_categorical(validlabels)
        print len(validlabels[0])
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

        model.save(saveas)
        return hist
    def test(self,testfile,validfile):
        model = load_model('calib48.h5')
        rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
        rawimg = rawimg[np.newaxis,...]
        predictions =  model.predict(rawimg)

        for prediction in predictions:
                totS = 0
                totY = 0
                totX = 0
                Z = np.sum(prediction > calib12Tresh)
                for pred,aclass in zip(prediction,adjclass):
                        if pred > calib12Tresh:
                                calib = adjclassV[adjclass.index(aclass)]
                                totS += calib[0]
                                totX += calib[1]
                                totY += calib[2]
                totS /= Z
                totX /= Z
                totY /= Z

        print "ACTUAL", adjclassV[tag]
        print "Predic", totS, totX, totY
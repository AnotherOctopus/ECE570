from keras.layers import Input, Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD,Adam
from keras import backend as K
from utils import * 
from config import *
import os
class calib12(object):
    def __init__(self):
            ## build model
            inp = Input(shape=(12,12,3),dtype='float32')
            x = Conv2D(16, (3, 3), strides = 1,padding="same")(inp)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size=(3,3),strides=2)(x)
            x = Flatten()(x)
            x = Dense(128)(x)
            x = Activation("relu")(x)
            x = Dense(45)(x)
            x = Activation("relu")(x)
            self.model = Model(inputs=inp,outputs=x)
            self.inp = inp
            self.out = x
    def loss(self,y_true,y_pred):
            K.print_tensor(y_true,message="y_true is")
            return (y_true-y_pred)**2
    def compile(self):

            self.model.compile(loss='binary_crossentropy',
                        optimizer=SGD(
                                lr=0.0001
                        ),
                        metrics=[categorical_accuracy])
    def train(self,saveas,train_data_dir,validation_data_dir,tags=["face","notface"]):

        # dimensions of images
        img_width, img_height = 12,12
        nb_train_samples = len(os.listdir(train_data_dir +"/face"))
        nb_validation_samples = len(os.listdir(validation_data_dir +"/face"))
        n_epochs = 1000
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        # data augmentation
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        trainimgs, trainlabels = collecttagsfromdir( train_data_dir, tag = tags[0])
        trainlabels = to_categorical(trainlabels)
        train_generator = train_datagen.flow(trainimgs,
                                            trainlabels)

        validimgs, validlabels = collecttagsfromdir(validation_data_dir, tag = tags[0])
        validlabels = to_categorical(validlabels)
        validation_generator = test_datagen.flow(validimgs,
                                                validlabels)
        # Train
        hist = self.model.fit(trainimgs,
                trainlabels,
                batch_size =64,
                epochs=n_epochs,
                validation_data=(validimgs,validlabels),
                shuffle=True,
                verbose=1)

        self.model.save("calib12.h5")
        return hist

    def test(self,testfile,validfile):
            model = load_model('calib12.h5')
            rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
            rawimg = rawimg[np.newaxis,...]

            with open(validfile,"r") as fh:
                    data = fh.read().strip('\n')
            predictions = model.predict(rawimg)
            for prediction in predictions:
                    totS = 0
                    totY = 0
                    totX = 0
                    Z = np.sum(prediction > CALIB12THRESH)
                    if Z == 0:
                            print prediction
                            print "Predic", 1,0,0
                            print "ACTUAL", adjclassV[adjclass.index(data)]
                            continue
                    for pred,aclass in zip(prediction,adjclass):
                            if pred > CALIB12THRESH:
                                    calib = adjclassV[adjclass.index(aclass)]
                                    totS += calib[0]
                                    totX += calib[1]
                                    totY += calib[2]
                    totS /= Z
                    totX /= Z
                    totY /= Z

                    print "Predic", totS, totX, totY
                    print "ACTUAL", adjclassV[adjclass.index(data)]
if __name__ == "__main__":
    c12 = calib12()
    c12.test("/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/faces/adj12/train/face/115.jpg",
             "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/faces/adj12/train/tag/115.txt"
    )
    # face is 0
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from net12_model import detect12


# dimensions of images
img_width, img_height = 12,12
# data
train_data_dir = "data/detect12/train"
validation_data_dir = "data/detect12/validation"
nb_train_samples = len(os.listdir(train_data_dir +"/face")) + len(os.listdir(train_data_dir +"/notface"))
nb_validation_samples = len(os.listdir(validation_data_dir +"/face")) + len(os.listdir(validation_data_dir +"/notface"))
n_epochs = 50
batch_size = 10


if __name__ == "__main__":
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
                                       shear_range=0.2,
                                       zoom_range=0.2,
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

    # Train
    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        verbose=1)

model.save("first_try.h5")

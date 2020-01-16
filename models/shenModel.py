from __future__ import absolute_import, division, print_function, unicode_literals

import os, csv, json

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

import keras
from keras.preprocessing.image import img_to_array, load_img
print("TensorFlow version is ", tf.__version__)
print("Keras version is ", keras.__version__)

import numpy as np
import pandas as pd


def image_network_train(learn_data_path):
    train_dir = os.path.join(learn_data_path, 'train')
    #validation_dir = os.path.join(learn_data_path, 'validation')
    validation_dir = os.path.join(learn_data_path, 'val')
    test_dir = os.path.join(learn_data_path, 'test')

    # calcucate the num of category
    num_category = 0
    for dirpath, dirnames, filenames in os.walk(train_dir):
        for dirname in dirnames:
            num_category += 1

    # All images will be resized to 299x299
    image_size = 299
    batch_size = 16

    # Rescale all images by 1./255 and apply image augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
                        train_dir,  # Source directory for the training images
                        target_size=(image_size, image_size),
                        batch_size=batch_size,
                        class_mode='categorical')

    # Flow validation images in batches of 20 using validation_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
                        validation_dir, # Source directory for the validation images
                        target_size=(image_size, image_size),
                        batch_size=batch_size,
                        class_mode='categorical')

    # Flow validation images in batches of 20 using test_datagen generator
    test_generator = test_datagen.flow_from_directory(
                        test_dir, # Source directory for the test images
                        target_size=(image_size, image_size),
                        batch_size=batch_size,
                        class_mode='categorical')

    # Create the base model from the pre-trained convnets
    IMG_SHAPE = (image_size, image_size, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = keras.applications.xception.Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    # Freeze the convolutional base
    base_model.trainable = False

    # モデル
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_category, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # early stopping
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.summary()

    # 更新される重みの数
    print('after', len(model.trainable_weights))

    # Train the model
    epochs = 30
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    test_steps = test_generator.n // batch_size

    history = model.fit_generator(train_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs=epochs,
                                  workers=4,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[es],
                                  class_weight={0:1.0, 1:0.4})

    loss, acc = model.evaluate_generator(validation_generator, steps=validation_steps)
    print('val loss: {}, val acc: {}'.format(loss, acc))

    # Fine tuning
    # Un-freeze the top layers of the model
    base_model.trainable = True

    # The nums of layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine tune from this layer onwards
    fine_tune_at = 108

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Compile the model using a much-lower training rate
    model.compile(optimizer = keras.optimizers.Adam(lr=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 更新される重みの数
    print('after Fine tune', len(model.trainable_weights))

    # Continue Train the model
    history_fine = model.fit_generator(train_generator,
                                       steps_per_epoch = steps_per_epoch,
                                       epochs=epochs,
                                       workers=4,
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps,
                                       callbacks=[es],
                                       class_weight={0:1.0, 1:0.4})


    # print(history_fine.history)
    model_val_acc = history_fine.history['val_accuracy'][-1]
    print('val_acc: ', model_val_acc)

    # save model into hdf5 file ----------
    model.save(learn_data_path + '/shen_model.h5')

    loss, acc = model.evaluate_generator(validation_generator, steps=validation_steps)
    print('val loss: {}, val acc: {}'.format(loss, acc))

    loss, acc = model.evaluate_generator(test_generator, steps=test_steps)
    print('Test loss: {}, Test acc: {}'.format(loss, acc))


if __name__ == '__main__':

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    data_dir = os.path.join(prj_root, "datasets")
    LEARN_PATH = data_dir
    image_network_train(LEARN_PATH)

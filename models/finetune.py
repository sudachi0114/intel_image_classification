
# 転移学習 - finetune

import os, sys
sys.path.append(os.pardir)

import time
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
#from utils.img_utils import inputDataCreator
from utils.model_handler import ModelHandler

# define -----
batch_size = 16
input_size = 224
channel = 3
target_size = (input_size, input_size)
input_shpe = (input_size, input_size, channel)
set_epochs = 40


def main():

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "datasets")

    # original train_data only or with_augmented data
    train_dir = os.path.join(data_dir, "train")
    # train_dir = os.path.join(data_dir, "train_with_aug")
    validation_dir = os.path.join(data_dir, "val")  # original validation data

    # pair of decreaced train_data and increased validation data
    # train_dir = os.path.join(data_dir, "red_train")
    # train_dir = os.path.join(data_dir, "train_with_aug")
    # validation_dir = os.path.join(data_dir, "validation")

    test_dir = os.path.join(data_dir, "test")


    # data load ----------
    data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = data_gen.flow_from_directory(train_dir,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   class_mode='categorical')
                                                           
    validation_generator = data_gen.flow_from_directory(validation_dir,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')
                                                           
    test_generator = data_gen.flow_from_directory(test_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  class_mode='categorical')

    data_checker, label_checker = next(train_generator)

    print("train data shape (in batch): ", data_checker.shape)
    print("train label shape (in batch): ", label_checker.shape)
    # print("validation data shape:", validation_data.shape)
    # print("validation label shape:", validation_label.shape)
    # print("test data shape:", test_data.shape)
    # print("test label shape:", test_label.shape)
    

    # build model ----------
    mh = ModelHandler(input_size, channel)

    # あとで重みの解凍をできるように base_model を定義
    base_model = mh.buildMnv1Base()
    base_model.trainable=False

    model = mh.addChead(base_model)

    model.summary()


    # instance EarlyStopping -----
    es = EarlyStopping(monitor='val_loss',
                       patience=5,
                       verbose=1,
                       restore_best_weights=True)


    print("\ntraining sequence start .....")
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")                

    start = time.time()
    print("\ntraining sequence start .....")

    # 準備体操 -----
    print("\nwarm up sequence .....")
    model.summary()
    _history = model.fit_generator(train_generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=set_epochs,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps,
                                   callbacks=[es],
                                   verbose=1)

    # fine tuning -----
    print("\nfine tuning.....")
    mh.setFineTune(base_model, model, 81)
    model.summary()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[es],
                                  verbose=1)

    elapsed_time = time.time() - start
    print( "elapsed time (for train): {} [sec]".format(time.time() - start) )


    print("\nevaluate sequence...")
    test_steps = test_generator.n // batch_size
    eval_res = model.evaluate_generator(test_generator,
                                        steps=test_steps,
                                        verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])


if __name__ == '__main__':
    main()

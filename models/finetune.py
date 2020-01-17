
# transfer learning with finetuning

import os, sys
sys.path.append(os.pardir)
import time
import numpy as np

"""
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
"""

import tensorflow as tf
import keras
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
sess = tf.Session(config=config)
K.set_session(sess)
print("TensorFlow version is ", tf.__version__)
print("Keras version is ", keras.__version__)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
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

    # use original data (but make validation data by oneself) -----
    train_dir = os.path.join(data_dir, "red_train")
    validation_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")


    """
    use_da_data = False
    increase_val = False
    print( "\nmode: Use Augmented data: {} | increase validation data: {}".format(use_da_data, increase_val) )

    # First define original train_data only as train_dir
    train_dir = os.path.join(data_dir, "train")
    if (use_da_data == True) and (increase_val == False):
        # with_augmented data (no validation increase)
        train_dir = os.path.join(data_dir, "train_with_aug")
    validation_dir = os.path.join(data_dir, "val")  # original validation data

    # pair of decreaced train_data and increased validation data
    if (increase_val == True):
        train_dir = os.path.join(data_dir, "red_train")
        if (use_da_data == True):
            train_dir = os.path.join(data_dir, "red_train_with_aug")
        validation_dir = os.path.join(data_dir, "validation")

    test_dir = os.path.join(data_dir, "test")

    print("\ntrain_dir: ", train_dir)
    print("validation_dir: ", validation_dir)
    """


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


    # confusion matrix -----
    print("\nconfusion matrix")
    pred = model.predict_generator(test_generator,
                                   steps=test_steps,
                                   verbose=3)

    test_label = []
    for i in range(test_steps):
        _, tmp_tl = next(test_generator)
        if i == 0:
            test_label = tmp_tl
        else:
            test_label = np.vstack((test_label, tmp_tl))    

    idx_label = np.argmax(test_label, axis=-1)  # one_hot => normal
    idx_pred = np.argmax(pred, axis=-1)  # 各 class の確率 => 最も高い値を持つ class
    
    cm = confusion_matrix(idx_label, idx_pred)
    print(cm)

    # Calculate Precision and Recall
    print(classification_report(idx_label, idx_pred))

    """
    # get return in dict-type
    print(classification_report(idx_label, idx_pred, output_dict=True))

    # get out evaluation report by each class ( with for scentence ) 
    class_nums = list( set(idx_label) )  # the `number` of class
    for cnum in class_nums:
          print(cnum,
                classification_report(idx_label, 
                                      idx_pred,
                                      output_dict=True)['{}'.format(cnum)]
          )
    # you can chose evaluate metrix by dict index like 
    print(classification_report(idx_label, idx_pred, output_dict=True)['accuracy'])
    """




if __name__ == '__main__':
    main()

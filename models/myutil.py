
# 自作 utility を用いて学習

import os, sys
sys.path.append(os.pardir)

import time
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

#from keras.preprocessing.image import ImageDataGenerator
from utils.img_utils import inputDataCreator
from utils.model_handler import ModelHandler

# define -----
batch_size = 16
input_size = 224
channel = 3
target_size = (input_size, input_size)
input_shpe = (input_size, input_size, channel)
set_epochs = 20


def main():

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "datasets")
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")


    # data load ----------
    train_data, train_label = inputDataCreator(train_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True)

    print("train data shape (in batch): ", train_data.shape)
    print("train label shape (in batch): ", train_label.shape)
    # print("validation data shape:", validation_data.shape)
    # print("validation label shape:", validation_label.shape)
    # print("test data shape:", test_data.shape)
    # print("test label shape:", test_label.shape)
    

    # build model ----------
    mh = ModelHandler()

    model = mh.buildTlearnModel(base='mnv2')

    model.summary()

    """

    print("\ntraining sequence start .....")
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")                

    start = time.time()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1)
    print( "elapsed time (for train): {} [sec]".format(time.time() - start) )


    print("\nevaluate sequence...")
    eval_res = model.evaluate_generator(test_generator,
                                        #batch_size=10,
                                        verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])
    """


if __name__ == '__main__':
    main()

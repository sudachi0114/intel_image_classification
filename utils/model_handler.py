
import os

from keras.models import Sequential
from keras.applications import VGG16, MobileNet, MobileNetV2, NASNetMobile, DenseNet121, Xception
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam

class ModelHandler:

    def __init__(self, input_size=224, channel=3, num_class=6):

        print( "\nModel Handler has instanced." )

        self.INPUT_SIZE = input_size
        self.CHANNEL = channel
        self.INPUT_SHAPE = (self.INPUT_SIZE, self.INPUT_SIZE, self.CHANNEL)  # ch_last

        self.BASE_MODEL_FREEZE = True

        self.CLASS_MODE = 'categorical'  # 'binary'
        print("class mode:", self.CLASS_MODE)

        self.NUM_CLASS = num_class


    def modelCompile(self, model, lr=1e-4):

        _lr = lr

        if self.CLASS_MODE == 'categorical':
            adapt_loss = 'categorical_crossentropy'
        elif self.CLASS_MODE == 'binary':
            adapt_loss = 'binary_crossentropy',

        model.compile(loss=adapt_loss,
                      optimizer=Adam(lr=_lr),
                      metrics=['accuracy'])

        print("set loss: ", adapt_loss)
        print("set lr = ", _lr)
        print( "\nyour model has compiled.\n" )

        return model


    def buildMyModel(self):

        print("build simple model...")

        model = Sequential()

        model.add(Conv2D(32, (3,3), activation='relu', input_shape=self.INPUT_SHAPE))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))

        if self.CLASS_MODE == 'categorical':
            model.add(Dense(self.NUM_CLASS, activation='softmax'))
        elif self.CLASS_MODE == 'binary':
            model.add(Dense(1, activation='sigmoid'))

        return self.modelCompile(model)

    
    def buildVgg16Base(self):

        print("building `VGG16` base...")

        # default INPUT_SIZE = 224
        base_model = VGG16(input_shape=self.INPUT_SHAPE,
                           weights='imagenet',
                           include_top=False)

        return base_model


    def buildMnv1Base(self):

        print("building `MobileNetV1` base model...")

        # default INPUT_SIZE = 224
        base_model = MobileNet(input_shape=self.INPUT_SHAPE,
                               weights='imagenet',
                               include_top=False)

        return base_model


    def buildMnv2Base(self):

        print("building `MobileNetV2` base model...")

        # default INPUT_SIZE = 224
        base_model = MobileNetV2(input_shape=self.INPUT_SHAPE,
                                 weights='imagenet',
                                 include_top=False)

        return base_model


    def buildNasNetMobileBase(self):

        print("building `NasNetMobile` base model...")

        # default INPUT_SIZE = 224
        base_model = NASNetMobile(input_shape=self.INPUT_SHAPE,
                                  weights='imagenet',
                                  include_top=False)

        return base_model


    def buildDenseNet121Base(self):

        print("building `DenseNet121` base model...")

        # default INPUT_SIZE = 224
        base_model = DenseNet121(input_shape=self.INPUT_SHAPE,
                                 weights='imagenet',
                                 include_top=False)

        return base_model


    def buildXceptionBase(self):

        print("building `Xception` base model...")

        # default INPUT_SIZE = 299
        base_model = Xception(input_shape=self.INPUT_SHAPE,
                              weights='imagenet',
                              include_top=False)

        return base_model



    # 転移 base に + 分類 head する (base model の重みはすべて凍結
    def buildTlearnModel(self, base='vgg16'):

        print("building Transifer-learn model...")
        model = Sequential()
        
        print("base model is {}".format(base))

        if base == 'vgg16':
            base_model = self.buildVgg16Base()
        elif base == 'mnv1':
            base_model = self.buildMnv1Base()
        elif base == 'mnv2':
            base_model = self.buildMnv2Base()
        elif base == 'nasnetmobile':
            base_model = self.buildNasNetMobileBase()
        elif base == 'densenet121':
            base_model = self.buildDenseNet121Base()
        elif base == 'xception':
            base_model = self.buildXceptionBase()

        print("attempt classifer head on base model.")

        model.add(base_model)

        if self.CLASS_MODE == 'categorical':
            model.add(GlobalAveragePooling2D())
            model.add(Dense(self.NUM_CLASS, activation='softmax'))
        elif self.CLASS_MODE == 'binary':
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

        # base_model のパラメータを凍結
        if self.BASE_MODEL_FREEZE:
            print("trainable weights before freeze: ", len(model.trainable_weights))
            base_model.trainable = False
            print("trainable weights after freeze: ", len(model.trainable_weights))


        return self.modelCompile(model)


    def addChead(self, base_model):

        print("attempt classifer head on base model.")

        model = Sequential()
        model.add(base_model)

        if self.CLASS_MODE == 'categorical':
            model.add(GlobalAveragePooling2D())
            model.add(Dense(self.NUM_CLASS, activation='softmax'))
        elif self.CLASS_MODE == 'binary':
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

        return self.modelCompile(model)


    def setFineTune(self, base_model, model, at, fine_lr=2e-5):

        print("execute prepare of fine-tuning")

        # Fine tuning
        # Un-freeze the top layers of the model
        base_model.trainable = True

        # The nums of layers are in the base model
        print("\nNumber of layers in the base model: ", len(base_model.layers))

        # Fine tune from this layer onwards
        fine_tune_at = at
        print( "un-freeze weights at {}: [{}]".format(fine_tune_at, base_model.layers[fine_tune_at].name) )

        # print("(base_model) trainable weights before freeze: ", len(base_model.trainable_weights))
        print("trainable weights before freeze: ", len(model.trainable_weights))

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        for i, layer in enumerate(base_model.layers):
            if layer.trainable:
                print("  trainable: {} | {}".format(i, layer.name))

        # print("(base_model) trainable weights after freeze: ", len(base_model.trainable_weights))
        print("trainable weights after freeze: ", len(model.trainable_weights))
        
        return self.modelCompile(model, lr=fine_lr)



if __name__ == '__main__':

    mh = ModelHandler()

    #model = mh.buildMyModel()
    #model = mh.buildVgg16Base()
    #model = mh.buildTlearnModel(base='mnv1')

    #model.summary()

    base_model = mh.buildMnv1Base()
    base_model.summary()

    model = mh.addChead(base_model)
    
    model.summary()

    mh.setFineTune(base_model, model, 81)

    model.summary()


import os
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
try:
    from img_utils import inputDataCreator
except:
    from utils.img_utils import inputDataCreator


class AugWithImgaug:

    def __init__(self, input_size=224, channel=3):

        # 最低限の dir 構成を保持
        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['prj_root'] = os.path.dirname(self.dirs['cwd'])
        self.dirs['data_dir'] = os.path.join(self.dirs['prj_root'], "datasets")

        # list of imgaug DA modes -----
        self.imgaug_aug_list = ['native',
                                'rotation',
                                'hflip',
                                'width_shift',
                                'height_shift',
                                'zoom',
                                'logcon',
                                'linecon',
                                'gnoise',
                                'lnoise',
                                'pnoise',
                                'flatten',
                                'sharpen',
                                'invert',
                                'emboss',  # 14
                                'someof',
                                #'plural'
        ]

        # attributes -----
        self.BATCH_SIZE = 10
        self.INPUT_SIZE = input_size
        self.CHANNEL = channel
        self.DO_SHUFFLE = True
        self.CLASS_MODE = 'binary'
        self.CLASS_LIST = ['NORMAL', 'PNEUMONIA']


    def img2array(self, target_dir, input_size, normalize=False):

        data, label = inputDataCreator(target_dir,
                                       input_size,
                                       normalize)
        return data, label

    
    def randomDataAugument(self, num_trans):
        # 以下で定義する変換処理の内ランダムに幾つかの処理を選択
        seq = iaa.SomeOf(num_trans, [
            iaa.Affine(rotate=(-90, 90), order=1, mode="edge"),
            iaa.Fliplr(1.0),
            iaa.OneOf([
                # 同じ系統の変換はどれか1つが起きるように 1つにまとめる
                iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge")
            ]),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge"),
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=[0.05 * 255, 0.2 * 255]),
                iaa.AdditiveLaplaceNoise(scale=[0.05 * 255, 0.2 * 255]),
                iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True)
            ]),
            iaa.OneOf([
                iaa.LogContrast((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0))
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.5, 1.0)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
            ]),
            iaa.Invert(1.0)
        ], random_order=True)

        return seq


    def imgaug_augment(self, target_dir, input_size, normalize=False, aug='native'):

        data, label = self.img2array(target_dir, input_size, normalize)

        if aug == 'native':
            return data, label
        elif aug == 'rotation':
            imgaug_aug = iaa.Affine(rotate=(-90, 90), order=1, mode="edge")  # 90度 "まで" 回転
        elif aug == 'hflip':
            imgaug_aug = iaa.Fliplr(1.0)  # 左右反転
        elif aug == 'width_shift':
            imgaug_aug = iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(左右)
        elif aug == 'height_shift':
            imgaug_aug = iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(上下)
            # imgaug_aug = iaa.Crop(px=(0, 40))  <= 平行移動ではなく、切り抜き
        elif aug == 'zoom':
            imgaug_aug = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge")  # 80~120% ズーム
            # これも keras と仕様が違って、縦横独立に拡大・縮小されるようである。
        elif aug == 'logcon':
            imgaug_aug = iaa.LogContrast((0.5, 1.5))
        elif aug == 'linecon':
            imgaug_aug = iaa.LinearContrast((0.5, 2.0))  # 明度変換
        elif aug == 'gnoise':
            imgaug_aug = iaa.AdditiveGaussianNoise(scale=[0.05*255, 0.2*255])  # Gaussian Noise
        elif aug == 'lnoise':
            imgaug_aug = iaa.AdditiveLaplaceNoise(scale=[0.05*255, 0.2*255])  # LaplaceNoise
        elif aug == 'pnoise':
            imgaug_aug = iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True)  # PoissonNoise
        elif aug == 'flatten':
            imgaug_aug = iaa.GaussianBlur(sigma=(0.5, 1.0))  # blur: ぼかし (平滑化)
        elif aug == 'sharpen':
            imgaug_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)) # sharpen images (鮮鋭化)
        elif aug == 'emboss':
            imgaug_aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))  # Edge 強調
        elif aug == 'invert':
            imgaug_aug = iaa.Invert(1.0)  # 色反転 <= これがうまく行かないので自分で作った。
        elif aug == 'someof':  # 上記のうちのどれか1つ
            imgaug_aug = iaa.SomeOf(1, [
                iaa.Affine(rotate=(-90, 90), order=1, mode="edge"),
                iaa.Fliplr(1.0),
                iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge"),
                iaa.LogContrast((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0)),
                iaa.AdditiveGaussianNoise(scale=[0.05*255, 0.25*255]),
                iaa.AdditiveLaplaceNoise(scale=[0.05*255, 0.25*255]),
                iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True),
                iaa.GaussianBlur(sigma=(0.5, 1.0)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                iaa.Invert(1.0)  # 14
            ])
        elif aug == 'plural':  # 異なる系統の変換を複数(1つの変換あとに画素値がマイナスになるとError..)
            imgaug_aug = self.randomDataAugument(2)
        else:
            print("現在 imgaug で選択できる DA のモードは以下の通りです。")
            print(self.imgaug_aug_list, "\n")
            raise ValueError("予期されないモードが選択されています。")

        aug_data = imgaug_aug.augment_images(data)
        aug_data = np.clip(aug_data, 0, 255)

        return aug_data, label



    def save_imgauged_img(self, targrt_dir, input_size, normalize=False, save_dir="prj_root", aug='rotation'):

        auged_data, label = self.imgaug_augment(target_dir=targrt_dir,
                                                input_size=input_size,
                                                normalize=normalize,
                                                aug=aug)
        if save_dir == "prj_root":
            self.dirs["save_dir"] = os.path.join(self.dirs['cnn_dir'], "chest_xray_auged_{}".format(aug))
        else:
            self.dirs["save_dir"] = save_dir
        os.makedirs(self.dirs["save_dir"], exist_ok=True)

        for j, class_name in enumerate(self.CLASS_LIST):
            idx = 0
            for i, data in enumerate(auged_data):
                if label[i] == j:
                    save_dir_each =  os.path.join(self.dirs["save_dir"], '{}'.format(class_name))
                    os.makedirs(save_dir_each, exist_ok=True)
                    save_file_cats = os.path.join(save_dir_each, "{}.{}.{}.jpg".format(class_name, aug, idx))

                    pil_auged_img = Image.fromarray(data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                    pil_auged_img.save(save_file_cats)
                    idx += 1


    def display_imgaug(self, target_dir, input_size, normalize=False, aug="rotation"):

        for n_confirm in range(3):  # 三回出力して確認
            print("{}回目の出力".format(n_confirm+1))
            self.DO_SHUFFLE = False
            data, label = self.imgaug_augment(target_dir,
                                              input_size,
                                              normalize,
                                              aug=aug)
            data = data / 255

            plt.figure(figsize=(12, 6))

            for i in range(10):
                plt.subplot(2, 5, i+1)
                plt.imshow(data[i])
                plt.title("l: [{}]".format(label[i]))
                plt.axis(False)

            plt.show()




if __name__ == '__main__':

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    data_dir = os.path.join(prj_root, "datasets")
    
    train_dir = os.path.join(data_dir, "train")
    red_train_dir = os.path.join(data_dir, "red_train")
    # validation_dir = os.path.join(data_dir, "validation")
    # test_dir = os.path.join(data_dir, "test")

    auger = AugWithImgaug()

    """
    train_data, train_label = auger.img2array(train_dir, 224, normalize=False)
    print(train_data.shape)
    print(train_label.shape)

    auged_data, label = auger.imgaug_augment(train_dir, 224, normalize=False, aug="invert")
    print(auged_data.shape)
    print(label.shape)
    """

    # auger.display_imgaug(train_dir, 224, normalize=False, aug="plural")

    mode = "train"  # "red_train"
    for i in range(2):

        if mode == "train":
            save_loc = os.path.join(data_dir, "auged_{}".format(i))
            target_dir = train_dir
        elif mode == "red_train":
            save_loc = os.path.join(data_dir, "red_auged_{}".format(i))
            target_dir = red_train_dir

        auger.save_imgauged_img(target_dir,
                                input_size=224,
                                save_dir=save_loc,
                                aug='plural')

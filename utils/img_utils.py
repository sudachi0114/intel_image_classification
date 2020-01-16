
# Iterator で返されるのが嬉しくないので、自家製 load_img を作成
import os, sys
sys.path.append(os.pardir)

import gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ignore_list = [".DS_Store"]

def load_img(fpath, array_size, ch='RGB'):
    """convert image file to numpy array by PIL

    # Args:
        fpath (str): 読み込みたいファイルのパス
        array_size (int): 画像読み込みの配列のサイズ (正方形を想定)
        ch (str): channel の数
            'RGB'  => 3 (default)
            'gray' => 1

    # Returns:
        img_array (np.ndarray): 画像を np.ndarray で読み込んだ配列
    """

    img_obj = Image.open(fpath)

    resized_img = img_obj.resize((array_size, array_size))
    resized_img = resized_img.convert('RGB')
    if ch == 'gray':
        resized_img = resized_img.convert('L')
    img_array = np.asarray(resized_img)

    return img_array



def loadImageFromDir(target_dir, input_size, normalize=False, ch='RGB'):
    """ディレクトリを指定して、その中にある画像を再帰的に読み込む

    # Args:
        target_dir (str): 読み込みたい画像が格納されているディレクトリ
        input_size (int): 各画像を読み込みたい配列のサイズ (正方形を想定)
            => これを load_img() の array_size に渡す
        ch (str): channel の数
            'RGB'  => 3 (default)
            'gray' => 1


    # Returns: img_arrays (np.ndarray): ディレクトリの中にあった画像をそれぞれ配列に変換して積み上げたもの
    """

    pic_list = os.listdir(target_dir)

    for fname in ignore_list:
        if fname in pic_list:
            pic_list.remove(fname)

    sorted_pic_list = sorted(pic_list)
    del pic_list
    print("found {} images ...".format(len(sorted_pic_list)))

    img_arrays = []
    for picture in sorted_pic_list:
        target = os.path.join(target_dir, picture)
        img_arr = load_img(target, input_size, ch)
        if normalize:
            img_arr = img_arr / 255.0
        img_arrays.append(img_arr)

    img_arrays = np.array(img_arrays)

    assert img_arrays.shape[0] == len(sorted_pic_list)
    gc.collect()

    return img_arrays


def inputDataCreator(target_dir, input_size, normalize=False, one_hot=False, ch='RGB'):
    """CNN などに入力する配列を作成する
        keras ImageDataGenerator の Iterator じゃない版

    # Args:
        target_dir (str): 画像データのディレクトリ
        input_size (int): 各画像を読み込みたい配列のサイズ (正方形を想定)
            => これを load_img() の array_size に渡す
        normalize (bool): 画像を読み込む際に [0, 1] に変換するか
            False => [0, 255] (default)
            True => [0, 1]
        one_hot (bool): label を one-hot 表現に変換する
            False => 0 or 1 (default)
            True => [1, 0] or [0, 1]
        ch (str): channel の数
            'RGB'  => 3 (default)
            'gray' => 1

    # Returns
        img_arrays (np.ndarray): 読み込んだ画像データの配列
        labels (np.ndarray): 読み込んだ画像に対する正解ラベル
    """

    class_list = os.listdir(target_dir)

    for fname in ignore_list:
        if fname in class_list:
            class_list.remove(fname)

    print("found {} classes ...".format(len(class_list)))

    img_arrays = []
    labels = []

    each_class_img_arrays = []
    label = []

    sorted_class_list = sorted(class_list)
    del class_list
    for class_num, class_name in enumerate(sorted_class_list):
        each_class_data_dir = os.path.join(target_dir, class_name)
        print("processing class {} as {} ".format(class_num, class_name), end="")

        each_class_img_arrays = loadImageFromDir(each_class_data_dir, input_size, normalize, ch)
        label = np.full(each_class_img_arrays.shape[0], class_num)

        if len(img_arrays) == 0:
            img_arrays = each_class_img_arrays
        else:
            img_arrays = np.vstack((img_arrays, each_class_img_arrays))

        if len(label) == 0:
            labels = label
        else:
            labels = np.hstack((labels, label))

    del each_class_img_arrays

    """ 場所を変えよう
    if normalize:
        print(sys.getsizeof(img_arrays))
        img_arrays = img_arrays / 255.0
        #pass
    """

    img_arrays = np.array(img_arrays)
    if ch == 'gray':
        img_arrays = np.expand_dims(img_arrays, axis=3)
    labels = np.array(labels)

    print("debug: ", labels[1])

    if one_hot:
        labels = np.identity(2)[labels.astype(np.int8)]

    assert img_arrays.shape[0] == labels.shape[0]

    gc.collect()

    return img_arrays, labels


def dataSplit(data, label, train_rate=0.6, validation_rate=0.3, test_rate=0.1, one_hot=True):
    """順番に積み重なっているデータに対して 2class の場合に等分割する関数    
     # Args:
        data (np.ndarray): 画像データの配列
        label (np.ndarray): 画像データの正解ラベルの配列
        train_rate (float): 全データにおける train data の割合 (0 ~ 1) で選択
                            (default: 0.6 == 60%)
        valiation_rate (float): 全データにおける validation data の割合 (0 ~ 1) で選択
                            (default: 0.2 == 20%)
        test_rate (float): 全データにおける test data の割合 (0 ~ 1) で選択
                            (default: 0.1 == 10%)
        one_hot (bool): 引数の label が one_hot 表現か否か
                            (default: True)
    # Return:
        train_data, train_label
        validation_data, validation_label
        test data, test_label
    """
    if one_hot:
        class_num = len(label[0])
    else:
        class_num = len(set(label))
    print("\nData set contain {} class data.".format(class_num))

    amount = data.shape[0]
    print("Data amount: ", amount)
    each_class_amount = int(amount / class_num)
    print("Data each class data amount: ", each_class_amount)
    
    train_data, train_label = [], []
    validation_data, validation_label = [], []
    test_data, test_label = [], []


    # calucurate each data size
    train_size = int( each_class_amount*train_rate )  # 300
    validation_size = int( each_class_amount*validation_rate )  # 150
    test_size = int( each_class_amount*test_rate )  # 50

    print("train_size: ", train_size)
    print("validation_size: ", validation_size)
    print("test_size: ", test_size)


    # devide data -----
    for i in range(class_num):
        each_class_data = []
        each_class_label = []

        if one_hot:
            # label の i番目が i である index を取得
            idx = np.where(label[:, i] == 1)
            # print("condition: ", condition)
        else:
            # i である label の index を取得
            idx = np.where(label == i)
            # print("idx: ", idx)
        each_class_label = label[idx]
        each_class_data = data[idx]        
        print("\nfound class {} data as shape: {}".format(i, each_class_data.shape))
        print("found class {} label as shape: {}".format(i, each_class_label.shape))


        # split data ----------
        each_train_data, each_validation_data, each_test_data = np.split(each_class_data,
                                                                         [train_size, train_size+validation_size])
        print("\ntrain_data at class{}: {}".format(i, each_train_data.shape))
        print("validation_data at class{}: {}".format(i, each_validation_data.shape))
        print("test_data at class{}: {}".format(i, each_test_data.shape))

        # 初回は代入, 2回目以降は (v)stack
        if len(train_data) == 0:
            train_data = each_train_data
        else:
            train_data = np.vstack((train_data, each_train_data))

        if len(validation_data) == 0:
            validation_data = each_validation_data
        else:
            validation_data = np.vstack((validation_data, each_validation_data))

        if len(test_data) == 0:
            test_data = each_test_data
        else:
            test_data = np.vstack((test_data, each_test_data))


        # split label ----------
        each_train_label, each_validation_label, each_test_label = np.split(each_class_label,
                                                                         [train_size, train_size+validation_size])
        print("\ntrain_label at class{}: {}".format(i, each_train_label.shape))
        print("validation_label at class{}: {}".format(i, each_validation_label.shape))
        print("test_label at class{}: {}".format(i, each_test_label.shape))
    
        # 初回は代入, 2回目以降は (v)stack
        if len(train_label) == 0:
            train_label = each_train_label
        else:
            if one_hot:
                train_label = np.vstack((train_label, each_train_label))
            else:
                train_label = np.hstack((train_label, each_train_label))

        if len(validation_label) == 0:
            validation_label = each_validation_label
        else:
            if one_hot:
                validation_label = np.vstack((validation_label, each_validation_label))
            else:
                validation_label = np.hstack((validation_label, each_validation_label))

        if len(test_label) == 0:
            test_label = each_test_label
        else:
            if one_hot:
                test_label = np.vstack((test_label, each_test_label))
            else:
                test_label = np.hstack((test_label, each_test_label))

        print("୨୧┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈୨୧")
    
    print("\n    ... end.\n")
    
    print("train_data.shape: ", train_data.shape)
    print("validation_data.shape: ", validation_data.shape)
    print("test_data.shape: ", test_data.shape)
    # print(test_label)

    # program test -----
    """ chest_xray data は class に対して不均衡なので、このチェックが効かない
    print("\ntest sequence... ")
    
    # train -----
    cls0_cnt = 0
    cls1_cnt = 0
    if one_hot:
        for i in range(len(train_label)):
            if train_label[i][0] == 1:
                cls0_cnt += 1
            elif train_label[i][1] == 1:
                cls1_cnt += 1
    else:
        cls0_cnt = len(train_label[train_label==0])
        cls1_cnt = len(train_label[train_label==1])
    assert cls0_cnt == cls1_cnt
    print("  -> train cleared.")

    # validation -----
    cls0_cnt = 0
    cls1_cnt = 0
    if one_hot:
        for i in range(len(validation_label)):
            if validation_label[i][0] == 1:
                cls0_cnt += 1
            elif validation_label[i][1] == 1:
                cls1_cnt += 1
    else:
        cls0_cnt = len(validation_label[validation_label==0])
        cls1_cnt = len(validation_label[validation_label==1])
    assert cls0_cnt == cls1_cnt
    print("  -> validation cleared.")
    
    # test -----
    cls0_cnt = 0
    cls1_cnt = 0
    if one_hot:
        for i in range(len(test_label)):
            if test_label[i][0] == 1:
                cls0_cnt += 1
            elif test_label[i][1] == 1:
                cls1_cnt += 1
    else:
        cls0_cnt = len(test_label[test_label==0])
        cls1_cnt = len(test_label[test_label==1])
    assert cls0_cnt == cls1_cnt
    print("  -> test cleared.\n")
    """

    return train_data, train_label, validation_data, validation_label, test_data, test_label



def display(img_array, label):

    plt.imshow(img_array)
    plt.title("label: {}".format(label))
    plt.show()



if __name__ == '__main__':

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "datasets")
    train_dir = os.path.join(data_dir, "train")
    train_normal_dir = os.path.join(train_dir, "NORMAL")
    train_normal_samples = os.listdir(train_normal_dir)[:10]
    single_sample = os.path.join(train_normal_dir, train_normal_samples[0])
    print("fname: ", single_sample)

    import argparse

    parser = argparse.ArgumentParser(description="画像読み込みに関する自家製ミニマルライブラリ (速度はあまりコミットしてないです..)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--display", type=int, default=99999)

    args = parser.parse_args()

    if args.test:
        """
        print("\ntesting load_img():")
        single_img_array = load_img(single_sample, 224)
        print("  result: ", single_img_array.shape)


        print("\ntesting loadImageFromDir():")
        train_data = loadImageFromDir(train_normal_dir, 224)
        print("  result: ", train_data.shape)
        """

        print("\ntesting inputDataCreator(train_data_dir, 224, normalize, one_hot):")
        data, label = inputDataCreator(train_dir,
                                       224,
                                       normalize=True,
                                       one_hot=True,
                                       ch='RGB')
        print("  result (data shape) : ", data.shape)
        print("    data: \n", data[0])
        print("  result (label shape): ", label.shape)
        print("    label: \n", label)


    if args.time:
        print("\ntesting loadImageFromDir() in large data:")
        class_list = ["NORMAL", "PNEUMONIA"]
        origin_dir = os.path.join(prj_root, "datasets", "train", class_list[1])

        import time
        start = time.time()

        large_test_datas = loadImageFromDir(origin_dir, 224)
        print("  result: ", large_test_datas.shape)

        print("elapsed time: ", time.time() - start, " [sec]")


    if args.display != 99999:
        print("read & display...")
        data, label = inputDataCreator(train_data_dir, 224)

        display(data[args.display], label[args.display])

    if args.split:
        flg = False
        data, label = inputDataCreator(train_dir,
                                       224,
                                       normalize=True,
                                       one_hot=flg)
        print(data.shape)
        print(label.shape)

        train_data, train_label, validation_data, validation_label, test_data, test_label = dataSplit(data,
                                                                                                      label,
                                                                                                      one_hot=flg)
        print(train_label.shape)
        print(train_label[0])

    print("Done.")



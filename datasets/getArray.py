
import os, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# define ----------
ignore_files = ['.DS_Store']

cwd = os.getcwd()

train_dir = os.path.join(cwd, "train")

# sheve class_list ----------
class_list = os.listdir(train_dir)
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)


train_0_dir = os.path.join(train_dir, class_list[0])
train_1_dir = os.path.join(train_dir, class_list[1])



# sample
print( os.path.join(train_0_dir, os.listdir(train_0_dir)[0]) )

def load_datas_from_dir(target_dir):

    start = time.time()

    img_list = []
    img_size = 224
    channel = 1
    img_shape = (img_size, img_size)

    print("\nloading data from: ", target_dir)
    target_list = sorted(os.listdir(target_dir))

    for i in range(len(target_list)):
        targf = os.path.join(target_dir, target_list[i])

        pil_obj = Image.open(targf)
        pil_obj = pil_obj.resize(img_shape)
        pil_obj = pil_obj.convert("L")
        # print(pil_obj)

        img_arr = np.array(pil_obj)
        # print(img_arr.shape)
        assert img_arr.shape == img_shape
        img_list.append(img_arr)

    img_list = np.array(img_list)
    print("  img_list shape: ", img_list.shape)
    print("  elaped time: {} [sec]".format(time.time() - start))


load_datas_from_dir(train_0_dir)
load_datas_from_dir(train_1_dir)

"""
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(img_list[i], cmap='gray')
    plt.axis(False)
    plt.title(i)
plt.show()
"""

rest_data_source = ["val", "test"]

for dname in rest_data_source:
    source_dir = os.path.join(cwd, dname)
    for i in range( len(class_list) ):
        dtarget = os.path.join(source_dir, class_list[i])
        load_datas_from_dir(dtarget)



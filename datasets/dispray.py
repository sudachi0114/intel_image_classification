
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# define ----------
ignore_files = ['.DS_Store']

cwd = os.getcwd()

train_dir = os.path.join(cwd, "train")

class_list = os.listdir(train_dir)
# sheve ----------
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)


train_0_dir = os.path.join(train_dir, class_list[0])
train_1_dir = os.path.join(train_dir, class_list[1])

validation_dir = os.path.join(cwd, "val")
validation_0_dir = os.path.join(validation_dir, class_list[0])
validation_1_dir = os.path.join(validation_dir, class_list[1])

test_dir = os.path.join(cwd, "test")
test_0_dir = os.path.join(test_dir, class_list[0])
test_1_dir = os.path.join(test_dir, class_list[1])

train_0_dir_list = sorted(os.listdir(train_0_dir))

print(train_0_dir_list[0])

img_list = []
for i in range(10):
    targf = os.path.join(train_0_dir, train_0_dir_list[i])

    pil_obj = Image.open(targf)
    #pil_obj.convert("L")
    #print(pil_obj)

    img_arr = np.array(pil_obj)
    print(img_arr.shape)
    img_list.append(img_arr)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(img_list[i], cmap='gray')
    plt.axis(False)
    plt.title(i)
plt.show()


def hankoki():
    train_1_dir_list = sorted(os.listdir(train_1_dir))

    print(train_1_dir_list[18])

    targf = os.path.join(train_0_dir, train_0_dir_list[18])

    pil_obj = Image.open(targf)
    pil_obj = pil_obj.resize((224, 224))
    #pil_obj.convert("L")
    #print(pil_obj)

    img_arr = np.array(pil_obj)
    print(img_arr.shape)

    plt.imshow(img_arr, cmap='gray')
    plt.axis(False)
    plt.show()

hankoki()


# imports
import os

# define
cwd = os.getcwd()

train_dir = os.path.join(cwd, "train")
class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)

red_train_dir = os.path.join(cwd, "red_train")
validation_dir = os.path.join(cwd, "validation")
test_dir = os.path.join(cwd, "test")


def countAmount(dir_name):

    for i in range(len(class_list)):
        sub_dir = os.path.join(dir_name, class_list[i])
        print(sub_dir)
        print("  └─ ", len(os.listdir(sub_dir)))



countAmount(train_dir)

print("--+--"*10)

countAmount(red_train_dir)

countAmount(validation_dir)

countAmount(test_dir)

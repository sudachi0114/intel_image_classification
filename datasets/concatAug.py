
import os, shutil

# train or red_train
mode = "train"  # "red_train"

# define
cwd = os.getcwd()
if mode == "train":
    train_dir = os.path.join(cwd, "train")
    save_loc = os.path.join(cwd, "train_with_aug")
elif mode == "red_train":
    train_dir = os.path.join(cwd, "red_train")
    save_loc = os.path.join(cwd, "red_train_with_aug")
os.makedirs(save_loc, exist_ok=True)


class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)



def copy(src_dir, file_list, dist_dir, param=None):

    for pic_name in file_list:
        copy_src = os.path.join(src_dir, pic_name)
        if param is not None:
            fname, ext = pic_name.rsplit('.', 1)
            fname = "{}_".format(param) + fname
            pic_name = fname + "." + ext
            copy_dst = os.path.join(dist_dir, pic_name)
        else:
            copy_dst = os.path.join(dist_dir, pic_name)
        shutil.copy(copy_src, copy_dst)


def main():

    # copy natural train data into concat directory -----
    for i, cname in enumerate(class_list):
        sub_train_dir = os.path.join(train_dir, cname)
        sub_train_list = os.listdir(sub_train_dir)
        print(sub_train_dir)
        print("get {} data".format(len(sub_train_list)))

        # make save concated data directory -----
        sub_save_loc = os.path.join(save_loc, cname)
        os.makedirs(sub_save_loc, exist_ok=True)

        print("copy.....")
        copy(sub_train_dir, sub_train_list, sub_save_loc)
        print("    Done.")



        # copy augmented data into concat directory -----
        for i in range(2):
            print("process aug_{} ----------".format(i))
            if mode == "train":
                auged_dir = os.path.join(cwd, "auged_{}".format(i))
            elif mode == "red_train":
                auged_dir = os.path.join(cwd, "red_auged_{}".format(i))

            sub_auged_dir = os.path.join(auged_dir, cname)
            sub_auged_list = os.listdir(sub_auged_dir)
            print(sub_auged_dir)
            print("get {} data".format(len(sub_auged_list)))

            print("copy.....")
            copy(sub_auged_dir, sub_auged_list, sub_save_loc, param=i)
            print("    Done.")



def check():

    print("\ncheck function has executed ...")
    print(save_loc)
    
    for cname in class_list:
        sub_auged_dir = os.path.join(save_loc, cname)
        print(sub_auged_dir)
        print("  data amount: ", len( os.listdir(sub_auged_dir) ) )


if __name__ == "__main__":
    # main()
    check()


import os, shutil

# define
cwd = os.getcwd()

train_dir = os.path.join(cwd, "train")
val_dir = os.path.join(cwd, "val")

class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)

# train の n 枚を validation に分け与える ( 分け与える枚数が class 毎均一の場合 )
#   validation_sep = n  # int で設定
# class0 の n 枚, class1 の m 枚 を validation に分け与える場合 (class で異なる)
# validation_sep = [n, m]
validation_sep = [150, 350]



def copy(src_dir, file_list, dist_dir, param=None):

    print( "copy from {} data".format(src_dir) )
    print( "  -> to {} .....".format(dist_dir) )
    print( "    amount: ", len(file_list) )

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

    print( "  Done." )



def main():

    print("re-distribute train_data and validation data")

    # increced validation data -----
    validation_dir = os.path.join(cwd, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    # decreased train data -----
    red_train_dir = os.path.join(cwd, "red_train")
    os.makedirs(validation_dir, exist_ok=True)


    for i, cname in enumerate(class_list):

        # make increaced validation and decreased train directory -----
        sub_validation_dir = os.path.join(validation_dir, cname)
        os.makedirs(sub_validation_dir, exist_ok=True)

        sub_red_train_dir = os.path.join(red_train_dir, cname)
        os.makedirs(sub_red_train_dir, exist_ok=True)


        # provide each data moving -----
        sub_train_dir = os.path.join(train_dir, cname)
        print("\n", sub_train_dir)

        sub_train_list = os.listdir(sub_train_dir)
        sub_train_amount = len(sub_train_list)
        print( "    └─ get {} data".format(sub_train_amount) )

        sub_train_list = sorted(sub_train_list)


        if type(validation_sep) == int:
            sep = sub_train_amount - validation_sep
            print("  give {} train data to validation data.".format(validation_sep))
        elif (type(validation_sep) == list) and (len(validation_sep) == 2):
            sep = sub_train_amount - validation_sep[i]
            print("  give {} train data to validation data.".format(validation_sep[i]))
        else:
            pass



        sub_red_train_list = sub_train_list[:sep]
        sub_inc_validation_list = sub_train_list[sep:]
        print("  => decreased train amount: ", len(sub_red_train_list))
        print("  => increased validation amount: ", len(sub_inc_validation_list), " +8")

        # file copy -----
        print("\n    !! execute distribution...\n" )
        copy(sub_train_dir, sub_red_train_list, sub_red_train_dir)
        copy(sub_train_dir, sub_inc_validation_list, sub_validation_dir)
        # copy original_val data => validation
        sub_val_dir = os.path.join(val_dir, cname)
        sub_val_list = os.listdir(sub_val_dir)
        copy(sub_val_dir, sub_val_list, sub_validation_dir)



def check():

    print("\ncheck function has executed ...")

    for check_target in ["red_train", "validation"]:
        for cname in class_list:
            target_dir = os.path.join(cwd, check_target, cname)

            print(target_dir)
            print("  data amount: ", len( os.listdir(target_dir) ) )


if __name__ == "__main__":
    # main()
    check()

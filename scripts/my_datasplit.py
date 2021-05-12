import os
import sys
import random


def split_by_ratio(root, cls_names, train_ratios, target_root):
    """
    Split the dataset by a constant ratio. Each class is a sub-folder.
    A class name list should be provided to find the sub-folder.

    Notice:
        1. Currently I adopt copy rather than move.
        2. Since each "file" under class path is maybe a dir (especially our MIL case),
            I use "cp -r" rather than "cp" or shutil.copy

    Args:
        root: (str)
        cls_names: (list of str)
        train_ratios: (float)
        target_root: (str)
    """
    ##check root
    train_dir = os.path.join(target_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    test_dir = os.path.join(target_root, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    info_file = os.path.join(target_root, "train_test_split.txt")
    # infos: [name, classname, train/test]
    train_infos = []
    test_infos = []
    for cls_name in cls_names:
        cls_path = os.path.join(root, cls_name)
        dir_list = [os.path.join(cls_path, x) for x in os.listdir(cls_path)]
        train_nums = len(dir_list) * train_ratios
        ##random sample
        train_list = random.sample(dir_list, int(train_nums))
        test_list = [x for x in dir_list if x not in train_list]

        print("Handling class {}\n".format(cls_name))
        print("Total nums: {}\n".format(len(dir_list)))
        print("Train nums: {}\n".format(len(train_list)))
        print("Test nums: {}\n".format(len(test_list)))

        train_target_dir = os.path.join(train_dir, cls_name)
        if not os.path.exists(train_target_dir):
            os.makedirs(train_target_dir)

        test_target_dir = os.path.join(test_dir, cls_name)
        if not os.path.exists(test_target_dir):
            os.makedirs(test_target_dir)

        for train_file in train_list:
            train_infos.append([train_file.split("/")[-1], cls_name, "train"])
            os.system("""cp -r "{}" "{}" """.format(train_file, train_target_dir))
            print("{} Done!\n".format(train_file))

        for test_file in test_list:
            test_infos.append([test_file.split("/")[-1], cls_name, "test"])
            os.system("""cp -r "{}" "{}" """.format(test_file, test_target_dir))
            print("{} Done!\n".format(test_file))

    with open(info_file, "w+") as f:
        for train_info in train_infos:
            f.write("{},{},{}\n".format(train_info[0], train_info[1], train_info[2]))

        for test_info in test_infos:
            f.write("{},{},{}\n".format(test_info[0], test_info[1], test_info[2]))


def split_kfolds(root, cls_names, n_folds, target_root):
    """
    split dataset into k folds

    Args:
        root: (str)
        cls_names: (list of str)
        n_folds: (int)
        target_root: (str)
    """
    folds_dir = [os.path.join(target_root, str(x)) for x in range(n_folds)]

    for idx, fold_dir in enumerate(folds_dir):
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        ##check root
        train_dir = os.path.join(fold_dir, "train")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir = os.path.join(fold_dir, "test")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        info_file = os.path.join(fold_dir, "train_test_split.txt")
        train_infos = []
        test_infos = []
        for cls_name in cls_names:
            cls_path = os.path.join(root, cls_name)
            dir_list = [os.path.join(cls_path, x) for x in os.listdir(cls_path)]
            dir_list.sort()
            train_nums = int(len(dir_list) / n_folds)
            ##random sample
            # for idx, fold_dir in enumerate(folds_dir):
            test_list = dir_list[idx * train_nums:(idx + 1) * train_nums]
            train_list = [x for x in dir_list if x not in test_list]

            print("Handling class {}\n".format(cls_name))
            print("Total nums: {}\n".format(len(dir_list)))
            print("Train nums: {}\n".format(len(train_list)))
            print("Test nums: {}\n".format(len(test_list)))

            train_target_dir = os.path.join(train_dir, cls_name)
            if not os.path.exists(train_target_dir):
                os.makedirs(train_target_dir)

            test_target_dir = os.path.join(test_dir, cls_name)
            if not os.path.exists(test_target_dir):
                os.makedirs(test_target_dir)

            for train_file in train_list:
                train_infos.append([train_file.split("/")[-1], cls_name, "train"])
                os.system("""cp -r "{}" "{}" """.format(train_file, train_target_dir))
                print("{} Done!\n".format(train_file))

            for test_file in test_list:
                test_infos.append([test_file.split("/")[-1], cls_name, "test"])
                os.system("""cp -r "{}" "{}" """.format(test_file, test_target_dir))
                print("{} Done!\n".format(test_file))

            with open(info_file, "w+") as f:
                for train_info in train_infos:
                    f.write("{},{},{}\n".format(train_info[0], train_info[1], train_info[2]))

                for test_info in test_infos:
                    f.write("{},{},{}\n".format(test_info[0], test_info[1], test_info[2]))
    #         train_list = random.sample(dir_list, min(int(train_nums), len(dir_list)))
    #         dir_list = [x for x in dir_list if x not in train_list]
    #
    #         print("Handling class {}\n".format(cls_name))
    #         print("Total nums: {}\n".format(len(dir_list)))
    #         print("One fold nums: {}\n".format(len(train_list)))
    #
    #         last_dir = os.path.join(fold_dir, cls_name)
    #         if not os.path.exists(last_dir):
    #             os.makedirs(last_dir)
    #
    #
    #         for train_file in train_list:
    #             train_infos.append([train_file.split("/")[-1], cls_name, str(idx)])
    #             os.system("""cp -r "{}" "{}" """.format(train_file, last_dir))
    #             print("{} Done!\n".format(train_file))
    #
    # with open(info_file, "w+") as f:
    #     for train_info in train_infos:
    #         f.write("{},{},{}\n".format(train_info[0], train_info[1], train_info[2]))


if __name__ == "__main__":
    root = "/home/tclin/Phase2/all"
    cls_names = ["pos", "neg"]
    train_ratios = 0.7
    target_dir = "/home/tclin/Phase2/5_folder"
    split_kfolds(root, cls_names, 5, target_dir)



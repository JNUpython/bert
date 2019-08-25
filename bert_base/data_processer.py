# -*- coding: utf-8 -*-
# @Time    : 2019/6/30 20:18
# @Author  : kean
# @Email   : ?
# @File    : data_processer.py
# @Software: PyCharm

import os
import pandas as pd
import re
from tqdm import tqdm
import tensorflow as tf


def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in tqdm(os.listdir(directory)):
        # print(file_path)
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


if __name__ == '__main__':
    # train_neg = load_directory_data(os.path.join("data/aclImdb_v1/aclImdb", "train", "neg"))
    train_pos = load_directory_data(os.path.join("data/aclImdb_v1/aclImdb", "train", "pos"))
    # print(train_neg[:3])
    # print(set(train_neg["sentiment"].values))  # {'2', '4', '1', '3'}
    print(set(train_pos["sentiment"].values))  # {'10', '7', '8', '9'}

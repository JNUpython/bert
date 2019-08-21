# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 23:39
# @Author  : kean
# @Email   : ?
# @File    : process_data.py
# @Software: PyCharm

import re
from collections import Counter
from mylogger import logger
import pandas as pd
from copy import copy
import collections
from tqdm import tqdm
import random

seed = 1233
random.seed(seed)

"""
经过统计发现，训练数据和bert的原有词典差异性不大，可以不用改变自定义的词典大小
"""


def count_train_data(file):
    df = pd.read_csv(file, encoding="utf-8", delimiter=",", header=0)
    logger.info(df.columns)
    reviews = df["Reviews"].values
    logger.info(len(reviews))
    words = []
    for line in reviews:
        line = line.strip()
        if line:
            logger.info(line)
            line = re.sub("[a-zA-Z]+", "@", line)
            line = re.sub("\d+", "&", line)
            line = re.sub("\s", "", line)
            words.extend(list(line))
    return Counter(words)


"""
将获取的一部分额外没有标注的电商评论数据用户pre-train bert
"""


def prepare_fine_tune_data():
    file = r"D:\projects_py\bert\data\zhejiang\goods_zh.txt"
    reviews = open(file, encoding="utf-8", mode="r").readlines()
    texts = []
    for line in reviews:
        line = line.split(",")[0].strip()
        line = re.sub("\s+|，+|。+", "，\n", line).strip()
        if len(line) >= 3:
            texts.append(line)

    open(r"D:\projects_py\bert\data\zhejiang\goods_zh_fine_tune.txt",
         encoding="utf8", mode="w").write("\n\n".join(texts))


"""
将任务作为一个序列化标注的问题，将数据整理为序列标注BIO
"""


def data_for_squence(input_file, output_file=None):
    df_reviews = pd.read_csv(input_file, encoding="utf-8", delimiter=",", header=0)
    reviews = df_reviews["Reviews"].values

    def sentence_clean(sentence):
        # 字母 数字
        sentence = re.sub("[a-zA-Z]+", "@", sentence)
        sentence = re.sub("\d+", "&", sentence)
        sentence = re.sub("\s", "", sentence)
        return sentence.strip()

    # 句子清洗
    sentences = list(map(sentence_clean, reviews))
    df_reviews["Reviews"] = sentences

    # 序列化文本
    if not output_file:
        # 句子序列化： test data
        sentences = list(map(list, sentences))
        f = lambda list_words: "\n".join(list_words)
        sentences = list(map(f, sentences))
        with open(r"D:\projects_py\bert\zhejiang\data\test.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(sentences))
            file.close()
    else:
        df_labels = pd.read_csv(output_file, encoding="utf-8", delimiter=",", header=0)
        logger.info(Counter(df_labels["Categories"].values))
        # print(df_reviews.info())
        # print(df_labels.info())
        text = ""
        for col_id, col_review in tqdm(df_reviews[["id", "Reviews"]].values):
            # logger.info(col_id)
            # logger.info(col_review)
            col_id_df = df_labels.loc[df_labels.id == col_id]
            # print(col_id_df)

            col_id_aspects = [v for v in col_id_df["AspectTerms"].values if "_" != v]
            # logger.info(col_id_aspects)
            col_review_label = col_review
            for v in col_id_aspects:
                # logger.info(v)
                if v:
                    v_replaced = "[B_at]" + "[I_at]" * (len(v) - 1)
                    col_review_label = re.sub(v, v_replaced, col_review_label, 1)
                # logger.info(col_review_label)

            col_id_opinions = [v for v in col_id_df["OpinionTerms"].values if "_" != v]
            for v in col_id_opinions:
                if v:
                    v_replaced = "[B_ot]" + "[I_ot]" * (len(v) - 1)
                    col_review_label = re.sub(v, v_replaced, col_review_label, 1)

            # logger.info(col_review_label)
            tmp = [v for v in re.split("\]|\[", col_review_label) if v]
            # logger.info(tmp)

            col_review_label = [[v] if v.endswith("t") else list(v) for v in tmp]
            # logger.info(col_review_label)
            # logger.info(col_review)
            tmp = []
            for v in col_review_label:
                tmp.extend(v)
            col_review_label = tmp
            # logger.info(tmp)

            col_review = list(col_review)
            logger.info(col_review)
            logger.info(col_review_label)
            try:
                assert (len(col_review_label) == len(col_review))
                # 其他地方已经进行过处理
                # col_review = ["[CLS]"] + col_review
                # col_review_label = ["C"] + col_review_label
                for k, v in zip(col_review, col_review_label):
                    v = v if v != k else "O"
                    print(k, v)
                    text += k + "\t" + v + "\n"
                text += "\n"

            except:
                logger.info(col_review)
                logger.info(col_review_label)
                # continue
                break

        text = text.strip().split("\n\n")
        random.shuffle(text)

        num_doc = len(text)
        split_index = int(num_doc * 0.2)

        text_dev = text[:split_index]
        text_train = text[split_index:]

        logger.info(len(text_dev))
        logger.info(len(text_train))

        with open(r"D:\projects_py\bert\zhejiang\data\dev.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(text_dev))
            file.close()

        with open(r"D:\projects_py\bert\zhejiang\data\train.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(text_train))
            file.close()


def count_category(output_file):
    df_labels = pd.read_csv(output_file, encoding="utf-8", delimiter=",", header=0)
    cates = df_labels.Categories.values
    len(cates)
    logger.info(Counter(cates))
    cates_ids = collections.OrderedDict()
    for v, k in enumerate(set(cates)):
        cates_ids[k] = v
    logger.info(cates)
    pd.Series(cates_ids).to_csv("./data/category_ids.csv")
    logger.info(cates_ids)
    return cates_ids


def data_for_squence2(input_file, output_file=None):
    """

    :param input_file:
    :param output_file:
    :return:
    """
    df_reviews = pd.read_csv(input_file, encoding="utf-8", delimiter=",", header=0)
    reviews = df_reviews["Reviews"].values

    def sentence_clean(sentence):
        # 字母 数字
        sentence = re.sub("[a-zA-Z]+", "@", sentence)
        sentence = re.sub("\d+", "&", sentence)
        sentence = re.sub("\s", "", sentence)
        return sentence.strip()

    # 句子清洗
    sentences = list(map(sentence_clean, reviews))
    df_reviews["Reviews"] = sentences

    # 序列化文本
    if not output_file:
        # 句子序列化： test data
        sentences = list(map(list, sentences))
        f = lambda list_words: "\n".join(list_words)
        sentences = list(map(f, sentences))
        with open(r"D:\projects_py\bert\zhejiang\data\test.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(sentences))
            file.close()
    else:
        df_labels = pd.read_csv(output_file, encoding="utf-8", delimiter=",", header=0)
        cates_id = count_category(output_file)
        # logger.info(Counter(df_labels["Categories"].values))
        # print(df_reviews.info())
        # print(df_labels.info())
        logger.info(cates_id)
        text = ""
        cols_name = "AspectTerms,A_start,OpinionTerms,O_start,Categories".split(",")
        for col_id, col_review in tqdm(df_reviews[["id", "Reviews"]].values):
            # logger.info(col_id)
            # logger.info(col_review)
            col_id_df = df_labels.loc[df_labels.id == col_id]
            # print(col_id_df)

            col_review = list(col_review)
            col_review_label = " ".join(col_review)  # 用空格进行分开
            # print(cols_name)
            for AspectTerms, A_start, OpinionTerms, O_start, Categories in col_id_df[cols_name].values:
                cate_id = cates_id.get(Categories)
                # logger.info(AspectTerms)
                # logger.info(OpinionTerms)
                if AspectTerms != "_":
                    suffix = "at_%d" % cate_id
                    A_replaced = "B" + "I" * (len(AspectTerms) - 1)
                    A_replaced = " ".join([v + "_" + suffix for v in A_replaced])
                    col_review_label = col_review_label.replace(" ".join(list(AspectTerms)), A_replaced)
                    # logger.info(col_review_label)

                if OpinionTerms != "_":
                    obf = "m"  # 修饰自身
                    try:
                        A_start = int(A_start)
                        O_start = int(O_start)
                        if A_start < O_start:
                            obf = "f"  # 修饰前面aspect
                        else:
                            obf = "b"  # 修饰后面aspect
                    except:
                        pass
                    suffix = "ot_%d_%s" % (cate_id, obf)
                    # logger.info(suffix)
                    O_replaced = "B" + "I" * (len(OpinionTerms) - 1)
                    O_replaced = " ".join([v + "_" + suffix for v in O_replaced])
                    # logger.info(O_replaced)
                    col_review_label = col_review_label.replace(" ".join(list(OpinionTerms)), O_replaced)
                    # logger.info(col_review_label)

            col_review_label = col_review_label.split(" ")
            # logger.info(col_review)
            # logger.info(col_review_label)

            try:
                assert (len(col_review_label) == len(col_review))
                # 其他地方已经进行过处理
                # col_review = ["[CLS]"] + col_review
                # col_review_label = ["C"] + col_review_label
                for k, v in zip(col_review, col_review_label):
                    v = v if v != k else "O"
                    # print(k, v)
                    text += k + "\t" + v + "\n"
                text += "\n"

            except:
                logger.info(col_review)
                logger.info(col_review_label)
                # continue
                break

        text = text.strip().split("\n\n")
        random.shuffle(text)

        num_doc = len(text)
        split_index = int(num_doc * 0.2)

        text_dev = text[:split_index]
        text_train = text[split_index:]

        logger.info(len(text_dev))
        logger.info(len(text_train))

        with open(r"D:\projects_py\bert\zhejiang\data\dev.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(text_dev))
            file.close()

        with open(r"D:\projects_py\bert\zhejiang\data\train.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(text_train))
            file.close()


def count_predcited_aspect_opinion():
    file = "./output/label_test.txt"
    with open(file, encoding="utf-8", mode="r") as file:
        ots = []
        ats = []
        flag = None
        word = ""
        for line in file.readlines():
            # logger.info(line)
            line = line.strip()
            items = re.split("\s+", line)
            if len(items) != 3:
                if word:
                    if flag == "at":
                        ats.append(word)
                        logger.info(word)
                    elif flag == "ot":
                        ots.append(word)
                        logger.info(word)
                    else:
                        raise Exception("bug")
                flag = None
                word = ""
                continue
            # logger.info(items)
            # 寻找目标词汇
            if line.endswith("B_at"):
                # 旧的目标
                if word:
                    if flag == "at":
                        ats.append(word)
                        logger.info(word)
                    elif flag == "ot":
                        ots.append(word)
                        logger.info(word)
                    else:
                        raise Exception("bug")
                # 新目标
                flag = "at"
                word = items[0]
                continue
            if word and flag == "at" and line.endswith("I_at"):
                # 寻找仅仅接着的
                word += items[0]
                continue
            if word and flag == "at" and not line.endswith("I_at"):
                ats.append(word)
                logger.info(word)
                flag = None
                word = ""

            if line.endswith("B_ot"):
                # 旧的目标
                if word:
                    if flag == "at":
                        ats.append(word)
                        logger.info(word)
                    elif flag == "ot":
                        ots.append(word)
                        logger.info(word)
                    else:
                        raise Exception("bug")
                # 新目标
                flag = "ot"
                word = items[0]
                continue
            if word and flag == "ot" and line.endswith("I_ot"):
                # 寻找仅仅接着的
                word += items[0]
                continue
            if word and flag == "ot" and not line.endswith("I_ot"):
                flag = None
                ots.append(word)
                logger.info(word)
                word = ""
        logger.info(ats)
        logger.info(ots)
        logger.info(Counter(ats))
        logger.info(Counter(ots))


if __name__ == '__main__':
    file_labels = r"D:\projects_py\bert\data\zhejiang\th1\TRAIN\Train_labels.csv"
    file_reviews = r"D:\projects_py\bert\data\zhejiang\th1\TRAIN\Train_reviews.csv"
    file_reviews_ = r"D:\projects_py\bert\data\zhejiang\th1\TEST\Test_reviews.csv"
    # train_count = count_train_data(file)
    # logger.info(train_count)
    # dict_words = open(r"D:\projects_py\bert\chinese_L-12_H-768_A-12\vocab.txt", encoding="utf-8").readlines()
    # dict_words = {v.strip() for v in dict_words if v.strip()}
    # logger.info(dict_words)
    # iter_keys = train_count.copy().keys()
    # for key in iter_keys:
    #     if key in dict_words:
    #         train_count.pop(key)
    # logger.info(train_count)
    # prepare_fine_tune_data()
    # data_for_squence(file_reviews, file_labels)
    data_for_squence(file_reviews, None)
    # count_predcited_aspect_opinion()
    # count_category(file_labels)
    data_for_squence2(file_reviews, file_labels)


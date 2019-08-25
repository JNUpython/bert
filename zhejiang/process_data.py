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


def sentence_clean(sentence):
    # 字母 数字
    sentence = re.sub("[a-zA-Z]+", "@", sentence)
    sentence = re.sub("\d+", "&", sentence)
    sentence = re.sub("\s", "", sentence)
    return sentence.strip()


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


def parse_ner_predict(predicted_file, category_ids_file):
    category_ids = {}
    for k, v in pd.read_csv(open(category_ids_file)).values:
        category_ids[v] = k
    logger.info(category_ids)
    assert (len(category_ids) != 0)

    # （1）评论ID（ID)：ID是每一条用户评论的唯一标识。
    #
    # （2）用户评论（Reviews）：用户对商品的评论原文。
    #
    # （3）属性特征词（AspectTerms）：评论原文中的商品属性特征词。例如“价格很便宜”中的“价格”。该字段结果须与评论原文中的表述保持一致。
    #
    # （4）观点词（OpinionTerms）：评论原文中，用户对商品某一属性所持有的观点。例如“价格很便宜”中的“很便宜”。该字段结果须与评论原文中的表述保持一致。
    #
    # （5）观点极性（Polarity）：用户对某一属性特征的观点所蕴含的情感极性，即负面、中性或正面三类。
    #
    # （6）属性种类（Category）：相似或同类的属性特征词构成的属性种类。例如“快递”和“物流”两个属性特征词都可归入“物流”这一属性种类
    res = []
    with open(predicted_file, encoding="utf-8", mode="r") as file:
        items = file.read().strip().split("\n\n")
        logger.info(len(items))

        # ID AspectTerms Opinions Polarities Categories
        patt = re.compile(". O O")

        for id, item in enumerate(items, 1):
            # 分析每个句子
            review = " ".join([line[0] for line in item.split("\n")])
            logger.info(review)
            ner_tokens_fake = [v.strip() for v in patt.split(item.strip()) if v.strip()]  # 这里还没有将标注的分开出来
            # logger.info(ner_tokens_fake)

            ner_tokens = []
            for fake in ner_tokens_fake:
                # 存在识别的结果存在连续现象
                fake_lines = fake.split("\n")  # ner识别群每个行
                token = []
                for line in fake_lines:
                    if line[4] == "B" and token:
                        # 以B字母为分割
                        ner_tokens.append("\n".join(token))
                        token = [line]
                    else:
                        token.append(line)
                if token:
                    ner_tokens.append("\n".join(token))

            # logger.info(ner_tokens)

            # 将token拆解成词汇 和 词汇第一个字符对应的标注信息
            word_info_pairs = []
            for token in ner_tokens:
                token_lines = token.split("\n")
                word = "".join([l[0] for l in token_lines])
                # 去第一个序列标注化后的结果提取信息
                if "at" in token_lines[0]:
                    info = token_lines[0][6:].split("_")
                else:
                    info = token_lines[-1][6:].split("_")
                word_info_pairs.append([word, info])

            # 解析结果到df的行
            ner_tokens_res = []
            for index, word_info in enumerate(word_info_pairs):
                word, info = word_info
                category = category_ids.get(int(info[1]))
                # logger.info(category)
                # logger.info(info)
                if info[0] == "at":
                    aspect = word
                    # aspect
                    opinion = None
                    if index > 0:
                        # 向前寻找修饰的情感词汇
                        former_word, former_info = word_info_pairs[index - 1]
                        if former_info[0] == "ot" and former_info[-1] == "b":
                            opinion = former_word
                    if not opinion and index < (len(word_info_pairs) - 1):
                        # 向后寻找修饰词汇
                        next_word, next_info = word_info_pairs[index + 1]
                        if next_info[0] == "ot" and next_info[-1] == "f":
                            opinion = next_word
                    row = [id, review, aspect, opinion, None, category, item]
                    ner_tokens_res.append(row)

                if info[0] == "ot" and info[-1] == "m":
                    opinion = word
                    row = [id, review, None, opinion, None, category, item]
                    ner_tokens_res.append(row)
            res.extend(ner_tokens_res)
            # break
        df = pd.DataFrame(data=res,
                          columns=["ID", "Review", "AspectTerms", "Opinions", "Polarities", "Categories", "Ner"])
        df.to_excel("./data/data_ner/ner_res.xlsx", index=False)


def data_for_sentimental():
    # test: 将序列化标注的test数据解析作为模型的输入，利用到序列化标注的结果
    columns = ["ID", "AspectTerms", "Opinions", "Polarities", "Categories", "Review"]
    path = r"D:\projects_py\bert\zhejiang\data_ner\ner_res.xlsx"
    df = pd.read_excel(path)
    df = df[columns].fillna(value="_")
    df.to_csv(r"D:\projects_py\bert\zhejiang\data_sentimental\test.csv", index=False)
    print(df[:3])

    # train：将训练数据对应的label opinion提取并作为序列化标注的结果
    df = pd.read_csv(open(r"D:\projects_py\bert\data\zhejiang\th1\TRAIN\Train_labels.csv", encoding="utf-8"), header=0)
    df = df[["id", "AspectTerms", "OpinionTerms", "Polarities", "Categories"]]
    sentiment_ids = collections.OrderedDict()
    for index, senti in enumerate(set(df["Polarities"].values)):
        sentiment_ids[senti] = index
    logger.info(sentiment_ids)
    pd.Series(sentiment_ids).to_csv(r"D:\projects_py\bert\zhejiang\data_sentimental\sentiment_ids.csv")
    df["Polarities"] = df["Polarities"].apply(lambda x: sentiment_ids[x.strip()])
    df.columns = columns[:-1]
    print(df[:3])
    # 给训练数据添加review
    df_review = pd.read_csv(open(r"D:\projects_py\bert\data\zhejiang\th1\TRAIN\Train_reviews.csv", encoding="utf8"),
                            header=0, index_col=["id"], dtype=str)
    # print(df_review[:3])
    f = lambda x: " ".join(list(sentence_clean(x)))
    df_review["Reviews"] = df_review["Reviews"].apply(f).values
    tmp = [df_review.loc[id]["Reviews"] for id in df["ID"].values]
    # logger.info(tmp)
    df["Review"] = tmp
    print(df_review[:3])

    indexes = list(range(len(df)))
    random.shuffle(indexes)
    df = df.iloc[indexes]
    num_row = len(df)
    split_index = int(num_row * 0.2)
    df_dev = df[:split_index]
    df_train = df[split_index:]
    logger.info(len(df_dev))
    logger.info(len(df_train))

    df_train.to_csv(r"D:\projects_py\bert\zhejiang\data_sentimental\train.csv", index=False)
    df_dev.to_csv(r"D:\projects_py\bert\zhejiang\data_sentimental\dev.csv", index=False)


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
    # data_for_squence(file_reviews_, None)
    # count_predcited_aspect_opinion()
    # count_category(file_labels)
    # data_for_squence2(file_reviews, file_labels)
    # file_predict = r"D:\projects_py\bert\zhejiang\data_ner\label_test.txt"
    # file_category_ids = r"D:\projects_py\bert\zhejiang\data_ner\category_ids.csv"
    # parse_ner_predict(file_predict, file_category_ids)
    data_for_sentimental()

# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 23:39
# @Author  : kean
# @Email   : ?
# @File    : process_data.py
# @Software: PyCharm

import re
from collections import Counter
from bert_base.mylogger import logger
import pandas as pd
from copy import copy
import collections
from tqdm import tqdm
import random
import numpy as np

seed = 1233
random.seed(seed)
import synonyms

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
            line = re.sub("\s|\.", "", line)
            words.extend(list(line))
    return Counter(words)


def prepare_fine_tune_data():
    """
    将获取的一部分额外没有标注的电商评论数据用户pre-train bert
    """
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


def sentence_clean(sentence):
    """
    将任务作为一个序列化标注的问题，将数据整理为序列标注BIO
    """
    # 字母 数字
    # sentence = re.sub("[a-zA-Z]+", "@", sentence)
    # sentence = re.sub("\d+", "&", sentence)
    # sentence = re.sub("\s", "", sentence)
    return sentence.strip()


def data_for_squence(input_file, output_file=None):
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


def count_category(output_file, data_dir):
    df_labels = pd.read_csv(output_file, encoding="utf-8", delimiter=",", header=0)
    cates = df_labels.Categories.values
    len(cates)
    logger.info(Counter(cates))
    cates_ids = collections.OrderedDict()
    for v, k in enumerate(sorted(list(set(cates)))):
        cates_ids[k] = v
    logger.info(cates)
    pd.Series(cates_ids).to_csv(data_dir + "/category_ids.csv")
    logger.info(cates_ids)
    return cates_ids


def data_for_squence2(input_file, output_file=None, data_dir="zhejiang/data_ner"):
    """
    NER识别的数据准备
    :param input_file:
    :param output_file:
    :return:
    """
    max_len = 0
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
        max_len = max([len(v) for v in sentences])

        with open(data_dir + "/test.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(sentences))
            file.close()
    else:
        df_labels = pd.read_csv(output_file, encoding="utf-8", delimiter=",", header=0)
        cates_id = count_category(output_file, data_dir)
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
            # logger.info(col_id_df)

            col_review = list(col_review)
            if len(col_review) > max_len:
                max_len = len(col_review)
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
                tmp = []
                for k, v in zip(col_review, col_review_label):
                    v = v if v != k else "O"
                    tmp.append(v)
                    # print(k, v)
                    text += k + "\t" + v + "\n"
                text += "\n"

                if sum([v == "O" for v in tmp]) == len(tmp):
                    raise Exception

            except:
                logger.info(col_review)
                logger.info(col_review_label)
                # continue
                logger.info("数据存在问题")
                break

        text = text.strip().split("\n\n")
        # random.shuffle(text)

        num_doc = len(text)
        split_index = int(num_doc * 0.2)

        text_dev = text[:split_index]
        text_train = text[split_index:]

        logger.info(len(text_dev))
        logger.info(len(text_train))

        with open(data_dir + "/dev.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(text_dev))
            file.close()

        with open(data_dir + "/train.txt", mode="w", encoding="utf-8") as file:
            file.write("\n\n".join(text_train))
            file.close()
    return max_len


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
    for k, v in pd.read_csv(open(category_ids_file), header=None).values:
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
        df.to_excel("./zhejiang/data_ner/ner_res.xlsx", index=False)


def data_for_sentimental():
    # test: 将序列化标注的test数据解析作为模型的输入，利用到序列化标注的结果
    columns = ["ID", "AspectTerms", "Opinions", "Polarities", "Categories", "Review"]
    path = "zhejiang/data_ner/ner_res.xlsx"
    df = pd.read_excel(path)
    df = df[columns].fillna(value="_")
    df.to_csv("./zhejiang/data_sentimental/test.csv", index=False)
    print(df[:3])

    # train：将训练数据对应的label opinion提取并作为序列化标注的结果
    df = pd.read_csv(open(r"D:\projects_py\bert\data\zhejiang\th1\TRAIN\Train_labels.csv", encoding="utf-8"), header=0)
    df = df[["id", "AspectTerms", "OpinionTerms", "Polarities", "Categories"]]
    sentiment_ids = collections.OrderedDict()
    for index, senti in enumerate(set(df["Polarities"].values)):
        sentiment_ids[senti] = index
    logger.info(sentiment_ids)
    pd.Series(sentiment_ids).to_csv("zhejiang/data_sentimental/sentiment_ids.csv")
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

    df_train.to_csv("zhejiang/data_sentimental/train.csv", index=False)
    df_dev.to_csv("zhejiang/data_sentimental/dev.csv", index=False)


def get_sentiment_result():
    import os
    path_data_sentimental = "/Users/mo/Documents/github_projects/zhijiang/JNU/bert/zhejiang/data_sentimental"
    df = pd.read_csv(open(os.path.join(path_data_sentimental, "sentiment_ids.csv"), encoding="GBK"), header=None)
    id_sentiment = {int(v): k for k, v in df.values}
    print(id_sentiment)
    df = pd.read_csv(open(os.path.join(path_data_sentimental, "test_results.tsv"), encoding="utf-8"), sep='\t',
                     header=None)
    # print(df)
    sentiment = [id_sentiment[np.argmax(three)] for three in df.values]
    print(sentiment)
    df = pd.read_csv(open(os.path.join(path_data_sentimental, "test.csv"), encoding="GBK"), header=0)
    df["Polarities"] = sentiment

    df_test = pd.read_csv(open(os.path.join(path_data_sentimental, "Test_reviews.csv"), encoding="utf-8"), header=0)

    df = pd.merge(left=df_test, right=df, left_on="id", right_on="ID", how="left")
    df = df.fillna(value="_")
    df.to_csv(os.path.join(path_data_sentimental, "ner_sentiment_res.csv"), index=False)
    df = df[["id", "AspectTerms", "Opinions", "Categories", "Polarities"]]
    df.to_csv(os.path.join(path_data_sentimental, "Result.csv"), encoding="utf-8", header=None, index=False)


def uniform():
    return random.uniform(0, 1)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


# def data_enforce(label_file, review_file):
#     """数据增强: 以0.3 的概率对样本进行替换"""
#     columns_1 = "id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities".split(",")
#     columns_2 = "id,Reviews".split(",")
#     df_labels = pd.read_csv(open(label_file, encoding="utf-8"), header=0)[columns_1]
#     df_reviews = pd.read_csv(open(review_file, encoding="utf-8"), header=0)[columns_2]
#     df_reviews.index = df_reviews["id"].values
#     print(df_labels[:3])
#     print(df_reviews[:3])
#     res_1 = []
#     res_2 = []
#     count = 0
#     # for _ in range(50):
#     for _ in range(3):  # test
#         print(_)
#         for row1 in df_labels.values:
#             count += 1
#             is_fake = False
#             row2 = df_reviews.loc[row1[0]].values
#             # print(row2)
#             # print(row1)
#             row_label = list(row1)
#             row_review = list(row2)
#             if row_label[1] != "_":
#                 # AspectTerms 随机替换
#                 aspect = row_label[1]
#                 # 对于置信度大于0.4 的均作为备选
#                 aspect_syn = [word for word, _ in zip(*synonyms.nearby(aspect)) if _ > 0.4]
#                 if uniform() < 0.3 and aspect_syn:
#                     # 随机选出一个替换
#                     aspect_replace = random.choice(aspect_syn)
#                     row_label[1] = aspect_replace
#                     row_review[1] = row_review[1].replace(aspect, aspect_replace)
#                     is_fake = True
#
#             if row_label[4] != "_":
#                 # 情感 随机替换
#                 opinion = row_label[4]
#                 # print(synonyms.nearby(opinion))
#                 opinion_syn = [word for word, _ in zip(*synonyms.nearby(opinion)) if _ > 0.4]
#                 if uniform() < 0.3 and opinion_syn:
#                     opinion_replace = random.choice(opinion_syn)
#                     row_label[4] = opinion_replace
#                     row_review[1] = row_review[1].replace(opinion, opinion_replace)
#                     is_fake = True
#             # 经过上面两次处理，被替换的概率低于0.49
#
#             if uniform() < 0.05:
#                 # 以较低的概率 对aspect 和opinion的位置进行交换
#                 if row_label[1] != "_" and row_label[4] != "_":
#                     # 对标签位置进行更改
#                     tmp = row_label[2]
#                     row_label[2] = row_label[5]
#                     row_label[5] = tmp
#                     tmp = row_label[3]
#                     row_label[3] = row_label[6]
#                     row_label[6] = tmp
#                     # 对文本进行更改
#                     row_review[1] = row_review[1] \
#                         .replace(row_label[1], row_label[4]) \
#                         .replace(row_label[4], row_label[1])
#                     is_fake = True
#
#             # 经过上面的操作被 变换的可能为低于0.54
#
#             if uniform() < 0.2:
#                 # 随机替换1-3个词汇个词汇
#                 seg_words = synonyms.seg(row_review[1].replace(row_label[1], "@").replace(row_label[4], "@"))[0]
#                 num = random_pick([1, 2, 3], [0.7, 0.25, 0.05])
#                 # print(seg_words, num)
#                 god_words = np.random.choice(seg_words, min(num, len(seg_words)), replace=False)
#                 for god_word in god_words:
#                     if god_word != "@" and god_word != "&":
#                         tmp = synonyms.nearby(god_word)[0]
#                         if tmp:
#                             # 同义词替换
#                             row_review[1] = row_review[1].replace(god_word, random.choice(tmp))
#                             is_fake = True
#
#             # 经过上面的处理， 该样本为生成样本的概率低于0.74
#
#             if uniform() < 0.1:
#                 # 随机交换两个词汇
#                 seg_words = synonyms.seg(row_review[1].replace(row_label[1], "@").replace(row_label[4], "@"))[0]
#                 if len(seg_words) > 5:
#                     tmp = np.random.choice(seg_words, 2, replace=False)
#                     row_review[1] = row_review[1].replace(tmp[0], tmp[1]).replace(tmp[1], tmp[0])
#                     is_fake = True
#
#             if uniform() < 0.05:
#                 # 随机删除一个字符
#                 char_index = random.randint(0, len(row_review[1]) - 1)
#                 if row_review[1][char_index] not in {v for v in (row_label[1] + row_label[4])}:
#                     row_review[1] = row_review[1][:char_index] + row_review[1][char_index + 1:]
#                     is_fake = True
#
#             # 经过前面的处理增强样本占比低于0.9
#             row_label[0] = count
#             row_review[0] = count
#
#             res_1.append(row_label + [is_fake])
#             res_2.append(row_review + [is_fake])
#     pd.DataFrame(data=res_1, columns=columns_1 + ["is_fake"]).to_csv("zhejiang/enforce_data/train_labels_enforce.csv",
#                                                                      index=False,
#                                                                      encoding="utf-8")
#     pd.DataFrame(data=res_2, columns=columns_2 + ["is_fake"]).to_csv("zhejiang/enforce_data/train_reviews_enforce.csv",
#                                                                      index=False,
#                                                                      encoding="utf-8")


def data_enforce_(label_file, review_file):
    """数据增强: 以0.3 的概率对样本进行替换"""
    columns_1 = "id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities".split(",")
    columns_2 = "id,Reviews".split(",")
    df_labels = pd.read_csv(open(label_file, encoding="utf-8"), header=0)[columns_1]
    df_reviews = pd.read_csv(open(review_file, encoding="utf-8"), header=0)[columns_2]
    print(df_labels[:3])
    print(df_reviews[:3])
    res_1 = []
    res_2 = []
    count = 0
    for _ in range(3):
    # for _ in range(3):  # test
        print(_)
        for row_re in df_reviews.values:
            count += 1
            # logger.info(count)
            is_fake = False
            change_type = ""

            # 随机选择一个label行进行更改
            rows_la = df_labels[df_labels.id == row_re[0]].values.copy()
            one_index = random.randint(0, len(rows_la) - 1)
            # print(rows_la)
            row_la = rows_la[one_index]
            # print(row_la)
            row_label = list(row_la)
            row_review = list(row_re)
            if row_label[1] != "_":
                # AspectTerms 随机替换
                aspect = row_label[1]
                # 对于置信度大于0.4 的均作为备选
                aspect_syn = [word for word, _ in zip(*synonyms.nearby(aspect)) if _ > 0.4]
                if uniform() < 0.5 and aspect_syn:
                    # 随机选出一个替换
                    aspect_replace = random.choice(aspect_syn)
                    row_label[1] = aspect_replace
                    row_review[1] = row_review[1].replace(aspect, aspect_replace)
                    is_fake = True
                    change_type += "+" + "替换aspect"

            if row_label[4] != "_":
                # 情感 随机替换
                opinion = row_label[4]
                # print(synonyms.nearby(opinion))
                opinion_syn = [word for word, _ in zip(*synonyms.nearby(opinion)) if _ > 0.4]
                if uniform() < 0.5 and opinion_syn:
                    opinion_replace = random.choice(opinion_syn)
                    row_label[4] = opinion_replace
                    row_review[1] = row_review[1].replace(opinion, opinion_replace)
                    is_fake = True
                    change_type += "+" + "替换opinion"

            # 经过上面两次处理，被替换的概率低于0.49

            if uniform() < 0.1:
                # 以较低的概率 对aspect 和opinion的位置进行交换
                if row_label[1] != "_" and row_label[4] != "_":
                    # 对标签位置进行更改
                    tmp = row_label[2]
                    row_label[2] = row_label[5]
                    row_label[5] = tmp
                    tmp = row_label[3]
                    row_label[3] = row_label[6]
                    row_label[6] = tmp
                    # 对文本进行更改
                    row_review[1] = row_review[1] \
                        .replace(row_label[1], row_label[4]) \
                        .replace(row_label[4], row_label[1])
                    is_fake = True
                    change_type += "+" + "交换item"

            # 经过上面的操作被 变换的可能为低于0.54

            if uniform() < 0.3:
                # 随机替换1-3个词汇个词汇
                seg_words = synonyms.seg(row_review[1].replace(row_label[1], "@").replace(row_label[4], "@"))[0]
                num = random_pick([1, 2, 3], [0.7, 0.25, 0.05])
                # print(seg_words, num)
                god_words = np.random.choice(seg_words, min(num, len(seg_words)), replace=False)
                for god_word in god_words:
                    if god_word != "@" and god_word != "&":
                        tmp = synonyms.nearby(god_word)[0]
                        if tmp:
                            # 同义词替换
                            row_review[1] = row_review[1].replace(god_word, random.choice(tmp))
                            is_fake = True
                change_type += "+" + "替换其他"

            # 经过上面的处理， 该样本为生成样本的概率低于0.74

            if uniform() < 0.1:
                # 随机交换两个词汇
                seg_words = synonyms.seg(row_review[1].replace(row_label[1], "@").replace(row_label[4], "@"))
                if len(seg_words) > 5:
                    tmp = np.random.choice(seg_words, 2, replace=False)
                    row_review[1] = row_review[1].replace(tmp[0], tmp[1]).replace(tmp[1], tmp[0])
                    is_fake = True
                    change_type += "+" + "交换其他词汇"

            if uniform() < 0.3:
                # 随机删除一个字符
                char_index = random.randint(0, len(row_review[1]) - 1)
                if row_review[1][char_index] not in {v for v in (row_label[1] + row_label[4])}:
                    row_review[1] = row_review[1][:char_index] + row_review[1][char_index + 1:]
                    is_fake = True
                    change_type += "+" + "删除"

            # 经过前面的处理增强样本占比低于0.9

            # 序列编号id
            rows_la[one_index] = np.array(row_label)
            for v in range(len(rows_la)):
                rows_la[v][0] = count
            row_review[0] = count
            # logger.info(rows_la)

            res_1.extend(rows_la)
            res_2.append(row_review + [is_fake, change_type, row_re[-1]])


    pd.DataFrame(data=res_1, columns=columns_1).to_csv("zhejiang/enforce_data/train_labels_enforce.csv",
                                                       index=False,
                                                       encoding="utf-8")
    pd.DataFrame(data=res_2, columns=columns_2 + ["is_fake", "change_type", "original_review"]).to_csv(
        "zhejiang/enforce_data/train_reviews_enforce.csv",
        index=False,
        encoding="utf-8")


if __name__ == '__main__':
    file_labels = r"data\zhejiang\th1\TRAIN\Train_labels.csv"
    file_reviews = r"data\zhejiang\th1\TRAIN\Train_reviews.csv"
    file_reviews_ = r"data\zhejiang\th1\TEST\Test_reviews.csv"
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
    file_predict = r"D:\projects_py\bert\zhejiang\data_ner\label_test.txt"
    file_category_ids = r"D:\projects_py\bert\zhejiang\data_ner\category_ids.csv"
    # parse_ner_predict(file_predict, file_category_ids)
    # data_for_sentimental()
    # get_sentiment_result()

    #  数据增强
    # data_enforce_(file_labels, file_reviews)  # 关闭seed
    file_labels = "zhejiang/enforce_data/train_labels_enforce.csv"
    file_reviews = "zhejiang/enforce_data/train_reviews_enforce.csv"
    file_reviews_ = "data/zhejiang/th1/TEST/Test_reviews.csv"
    m1 = data_for_squence2(file_reviews_, None, data_dir="zhejiang/data_ner_enforce")  # 开启seed
    m2 = data_for_squence2(file_reviews, file_labels, data_dir="zhejiang/data_ner_enforce")  # 开启seed
    print(m1, m2)

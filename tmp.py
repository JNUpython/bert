# -*- coding: utf-8 -*-
# @Time    : 2019/8/30 10:07
# @Author  : kean
# @Email   : ?
# @File    : tmp.py
# @Software: PyCharm


"""
修复有category文件不一致导致的错误
"""

import pandas as pd

# df = pd.read_csv(open("zhejiang/data_sentimental/Result_error_cate.csv", encoding="utf-8"), header=None)
# values = df.values
#
# cate_id_error = {k: v for k, v in pd.read_csv("zhejiang/data_ner/category_ids.csv", header=None, encoding="GBK").values}
# print(cate_id_error)
#
# id_cate_right = {k : v for v, k in pd.read_csv("zhejiang/data_ner_enforce/category_ids.csv", header=None, encoding="GBK").values}
# print(id_cate_right)
# for index in range(len(values)):
#     if values[index][3] != "_":
#         values[index][3] = id_cate_right[cate_id_error[values[index][3]]]
#
#
# pd.DataFrame(data=values).to_csv("zhejiang/data_sentimental/Result.csv", encoding="utf-8", index=False)
#
# lines_8 = open(r"zhejiang/data_ner_enforce/label_test8500.txt", encoding="utf-8", mode="r").readlines()
# lines_10 = open(r"zhejiang/data_ner_enforce/label_test10500.txt", encoding="utf-8", mode="r").readlines()
#
# count = 0
# for v1, v2 in zip(lines_8, lines_10):
#     count += 1
#     if not v1.strip():
#         print("+++++++++++++++++")
#     if v1 != v2:
#         print(count)
#         print(v1.strip() + " vs " +  v2)

# df_1 = pd.read_csv(open("zhejiang/data_sentimental/epoch10500_旧情感模型/ner_sentiment_res旧情感模型epoch10050.csv"), header=0)
# print(df_1[:3])
# df_2 = pd.read_csv("zhejiang/data_sentimental/Result0829_zh.csv", encoding="utf-8", header=None)
# df_2.columns = ["旧的" + i for i in ["id", "aspect", "opinion", "category", "sentiment"]]
# print(df_2[:3])
#
# df = pd.merge(left=df_1, right=df_2, left_on="id", right_on="旧的id", how="outer")
# df.to_excel("tmp.xlsx", index=False)

df = pd.read_csv("zhejiang/data_sentimental/Test_reviews.csv", encoding="utf-8")
import re

count = 0
ids = set()
for row in pd.read_csv("zhejiang/data_sentimental/Result0831_zh.csv", encoding="utf-8").values:
    count += 1
    index = row[0]
    ids.add(index)
    aspect = row[1]
    opinion = row[2]
    review = df[df.id == index].values[0][1]
    review = re.sub("/s+", "，", review)
    if aspect != "_" and aspect not in review:
        print(count, index)
        print(aspect, review)

    if  opinion != "_" and opinion not in review:
        print(count, index)
        print(opinion, review)

print(set(ids) == set(df.id.values))






# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 22:20
# @Author  : kean
# @Email   : ?
# @File    : data_file_format.py
# @Software: PyCharm


from mylogger import logger

"""
将数据准备成标准的文件
"""

def cola(infile, outfile):
    file_read = open(infile, encoding="utf-8", mode="r")
    sentences = []
    for line in file_read.readlines():
        line = line.strip()
        sentences.append(line)
        # cols = line.split("\t")
        # assert len(cols) == 4
        # label + sentence
        # sentences.append(cols[1] + "\t" + cols[-1])
    # file_read.closed()
    file_write = open(outfile, encoding="utf-8", mode="w")
    file_write.write("\n".join(sentences))
    # file_write.closed()
    logger.info("样本数据num：%d" % len(sentences))



if __name__ == '__main__':
    cola("data/cola_public/raw/in_domain_dev.tsv", "data/cola_public/format/dev.tsv")
    cola("data/cola_public/raw/in_domain_train.tsv", "data/cola_public/format/train.tsv")
    cola("data/cola_public/raw/out_of_domain_dev.tsv", "data/cola_public/format/test.tsv")
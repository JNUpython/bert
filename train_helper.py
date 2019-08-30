# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/30 14:01
 @Author  : MaCan (ma_cancan@163.com)
 @File    : train_helper.py
"""

import argparse
import os

__all__ = ['get_args_parser']


def get_args_parser():
    parser = argparse.ArgumentParser()
    # if os.name == 'nt':
    #     bert_path = 'F:\chinese_L-12_H-768_A-12'
    #     root_path = r'C:\workspace\python\BERT-BiLSTM-CRF-NER'
    # else:
    #     bert_path = '/home/macan/ml/data/chinese_L-12_H-768_A-12/'
    #     root_path = '/home/macan/ml/workspace/BERT-BiLSTM-CRF-NER'

    # 第一组参数
    group1 = parser.add_argument_group('File Paths',
                                       'config the path, checkpoint and filename of a pretrained/fine-tuned BERT model')
    group1.add_argument('-data_dir', type=str,
                        default="./zhejiang/data_ner",
                        help='train.csv, dev.csv and test.csv 数据文件存放路径')

    group1.add_argument('-bert_config_file', type=str,
                        default=r"D:\projects_py\bert\chinese_L-12_H-768_A-12\bert_config.json")

    group1.add_argument('-output_dir', type=str,
                        default=r"./zhejiang/output_ner",
                        help='模型训练后保存路径')

    group1.add_argument('-init_checkpoint', type=str,
                        default=r"./chinese_L-12_H-768_A-12",  # 初始bert模型
                        # default=r"D:\projects_py\bert\zhejiang\output",  # 训练后的模型
                        help='Initial checkpoint (usually from a pre-trained BERT model).')

    group1.add_argument('-vocab_file', type=str,
                        default=r"./chinese_L-12_H-768_A-12/vocab.txt")

    # 第二组关于模型的一些参数
    group2 = parser.add_argument_group('Model Config', 'config the model params')
    group2.add_argument('-max_seq_length', type=int,
                        default=128,
                        # default=64,
                        help='输入序列的允许最大长度，即句子 tokens 的长度')

    group2.add_argument('-do_train', action='store_false',
                        default=True,
                        # default=False,
                        help='Whether to run training.')

    group2.add_argument('-do_eval', action='store_false',
                        default=False,
                        # default=False,
                        help='Whether to run eval on the dev set.')

    group2.add_argument('-do_predict', action='store_false',
                        default=True,
                        help='Whether to run the predict in inference mode on the test set.')

    group2.add_argument('-batch_size', type=int,
                        # default=1,  #  for test
                        default=32,
                        help='Total batch size for training, eval and predict.')

    group2.add_argument('-learning_rate', type=float,
                        default=1e-5,
                        help='The initial learning rate for Adam.')

    group2.add_argument('-num_train_epochs', type=float,
                        default=10,
                        help='Total number of training epochs to perform.')

    group2.add_argument('-dropout_rate', type=float,
                        default=0.5,
                        # default=0.0,
                        help='Dropout rate')

    group2.add_argument('-clip', type=float,
                        default=0.5,
                        help='Gradient clip')

    group2.add_argument('-warmup_proportion', type=float,
                        default=0.1,
                        help='Proportion of training to perform linear learning rate warmup for '
                             'E.g., 0.1 = 10% of training.')

    group2.add_argument('-lstm_size', type=int,
                        default=None,
                        help='size of lstm units.')

    group2.add_argument('-num_layers', type=int,
                        default=0,
                        help='number of rnn layers, default is 1.')

    group2.add_argument('-cell', type=str,
                        default='lstm',
                        help='which rnn cell used.')

    group2.add_argument('-save_checkpoints_steps', type=int,
                        default=500,
                        help='save_checkpoints_steps')

    group2.add_argument('-save_summary_steps', type=int,
                        default=10,
                        help='save_summary_steps.')

    group2.add_argument('-filter_adam_var', type=bool,
                        default=False,
                        help='训练完之后是否删除adam的参数，不存储在model中')

    group2.add_argument('-do_lower_case', type=bool,
                        default=True,
                        help='Whether to lower case the input text.')

    group2.add_argument('-clean', type=bool,
                        default=False,
                        # default=True，
                        help="是否清除output路径下面文件， 继续训练模型请设置为False")

    group2.add_argument('-device_map', type=str,
                        # default='0',  # GPU
                        default='-1',  # CPU
                        help='witch device using to train')

    # add labels
    group2.add_argument('-label_list', type=str,
                        default=None,
                        # default=r"/output_ner/label_list.pkl",
                        help='User define labels， can be a file with one label one line or a string using \',\' split')

    parser.add_argument('-verbose', action='store_true',
                        default=False,
                        help='turn on tensorflow logging for debug')

    parser.add_argument('-ner', type=str,
                        default='ner',
                        help='which modle to train')

    return parser.parse_args()

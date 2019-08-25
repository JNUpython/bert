# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 14:58
# @Author  : kean
# @Email   : ?
# @File    : bert_sentimental_flags.py
# @Software: PyCharm

import tensorflow as tf

flags = tf.flags
# flags.

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_dir",
                    r"D:\projects_py\bert\zhejiang\data_sentimental",
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")
flags.DEFINE_string("bert_config_file",
                    r"D:\projects_py\bert\chinese_L-12_H-768_A-12\bert_config.json",
                    "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
flags.DEFINE_string("task_name",
                    "zhejiang",
                    "The name of the task to train.")
flags.DEFINE_string("vocab_file",
                    r"D:\projects_py\bert\chinese_L-12_H-768_A-12\vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("output_dir",
                    r"D:\projects_py\bert\zhejiang\output_sentiment",
                    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string("init_checkpoint",
                    r"D:\projects_py\bert\chinese_L-12_H-768_A-12",
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case",
                  True,
                  "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

flags.DEFINE_integer("max_seq_length",
                     64,  # 128 内存爆掉
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded.")
flags.DEFINE_bool("do_train",
                  True,
                  "Whether to run training.")

flags.DEFINE_bool("do_eval",
                  True,
                  "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict",
                  True,
                  "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size",
                     32,
                     "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size",
                     32,
                     "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size",
                     32,
                     "Total batch size for predict.")

flags.DEFINE_float("learning_rate",
                   5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs",
                   40,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion",
                   0.1,
                   "Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps",
                     50,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop",
                     50,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu",
                  False,
                  "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string("tpu_name",
                       None,
                       "The Cloud TPU to use for training. This should be either the name "
                       "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

tf.flags.DEFINE_string("tpu_zone",
                       None,
                       "[Optional] GCE zone where the Cloud TPU is located in. If not "
                       "specified, we will attempt to automatically detect the GCE project from "
                       "metadata.")

tf.flags.DEFINE_string("gcp_project",
                       None,
                       "[Optional] Project name for the Cloud TPU-enabled project. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string("master",
                       None,
                       "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores",
                     8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")


# print(FLAGS)


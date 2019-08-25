# -*- coding: utf-8 -*-
# @Time    : 2019/8/18 10:02
# @Author  : kean
# @Email   : ?
# @File    : run.py
# @Software: PyCharm


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from bert_base.mylogger import logger

from train_helper import get_args_parser
from bert_lstm_crf import train


def train_ner():
    args = get_args_parser()
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        logger.info('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    logger.info(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    train(args=args)


if __name__ == '__main__':
    train_ner()

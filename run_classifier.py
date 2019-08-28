# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import bert_base.modeling as modeling
import bert_base.optimization as optimization
import bert_base.tokenization as tokenization
import tensorflow as tf
from bert_base.mylogger import logger, print_falgs
from sentiment_flags import FLAGS
from tqdm import tqdm
from self_attention import SelfAttention2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class InputExample:
    """输入样本"""

    def __init__(self, guid, text_a, text_b=None, label=None, aspect=None, opinion=None):
        """Constructs a InputExample.
          guid: ID
          text_a: string. The untokenized text of the first sequence. For single sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence. Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.aspect = aspect
        self.opinion = opinion

    def __str__(self):
        return "InputExample: %s" % "\n".join(
            [str(v) for v in [self.guid, self.text_a, self.text_b, self.label, self.aspect, self.opinion]])


class PaddingInputExample:
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, Q_mask, is_real_example=True):
        # 输入的文本id序列
        self.input_ids = input_ids
        # 对应的序列是否被 mask
        self.input_mask = input_mask
        # 如果是多个句子组成，需要利用此序列标注进行分割
        self.segment_ids = segment_ids
        # 序列化标注的结果，或者分类的结果
        self.label_id = label_id
        self.Q_mask = Q_mask
        # ??
        self.is_real_example = is_real_example


class DataProcessor:
    """
    数据预处理的抽象函数
    """

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        raise NotImplementedError()


class ZhejiangProcesser(DataProcessor):
    """读取枝江的情感分析分析"""

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        logger.info("_read_tsv file: %s" % input_file)
        reader = csv.reader(open(input_file, encoding="GBK", mode="r"), delimiter=",", quotechar=quotechar)
        lines = []
        for line in reader:
            # logger.info(line)
            lines.append(line)
        # for test：测试模型是否通
        # lines = lines[:32]
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            # Only the test set has a header
            # logger.info(line[0])
            guid = "%s-%s" % (set_type, i)
            # 原始数据不需要经过处理，直接选取1， 3 列
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[-1])
                label = "0"

            else:
                text_a = tokenization.convert_to_unicode(line[-1])
                label = tokenization.convert_to_unicode(line[3])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=line[1], opinion=line[2]))
            # logger.info(examples[0])
            # break
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
    将一个InputExample 计算序列化 转化为 InputExample
    :param ex_index:
    :param example:
    :param label_list:
    :param max_seq_length:
    :param tokenizer:
    :return:
    """

    if isinstance(example, PaddingInputExample):
        # 最后一个batch可能存在数量不够用 空的来充数
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            Q_mask=[0] * max_seq_length,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 计算 Q_mask
    Q_mask = "\t".join(tokens)
    if example.aspect:
        aspect = "\t".join(list(example.aspect))
        Q_mask = Q_mask.replace(aspect, "\t".join(["mask"] * len(example.aspect)))

    if example.opinion:
        opinion = "\t".join(list(example.opinion))
        Q_mask = Q_mask.replace(opinion, "\t".join(["mask"] * len(example.opinion)))
    Q_mask = [1 if v == "mask" else 0 for v in Q_mask.split("\t")]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        Q_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(Q_mask) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 3:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("Q_mask: %s" % " ".join([str(x) for x in Q_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("label: %s (id = %d)" % (example.label, label_id))


    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        Q_mask=Q_mask,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """
    将序列化后的InputFeature 转化为 tfrecord
    :param examples:
    :param label_list:
    :param max_seq_length:
    :param tokenizer:
    :param output_file:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    logger.info("数据准备....")
    for (ex_index, example) in tqdm(enumerate(examples)):
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        features["Q_mask"] = create_int_feature(feature.Q_mask)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    logger.info("数据准备完成！")


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "Q_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels,
                 use_one_hot_embeddings, Q_mask,
                 p_coef=0.004  # 注意机制的惩罚系数
                 ):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        Q_mask=Q_mask
    )

    # In the demo, we are doing a simple classification task on the entire segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.

    output_layer = model.get_sequence_output()  # 隐藏层最后的输出
    logger.info(modeling.get_shape_list(output_layer))
    output_layer2 = model.get_sequence_output2()
    logger.info(modeling.get_shape_list(output_layer2))

    if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer2 = tf.nn.dropout(output_layer2, keep_prob=0.9)

    # 添加一个住一层输出机制
    # 这里几个参数 d_a 是超参数，代表网络的复杂程度，就像网络的层数一样
    # fc 全连接的层的节点的大小
    # r_size 代表对句子embedding的维度的序列长度大小
    aspect_opinion_attention = SelfAttention2(output_layer, output_layer2, num_classes=num_labels,
                                              d_a_size=64,
                                              r_size=10,
                                              fc_size=10
                                              )
    logits, panel = aspect_opinion_attention.get_output()

    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    # 每个样本的交叉信息熵
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # 求均值损失 + 注意的损失惩罚
    loss = tf.reduce_mean(per_example_loss) + tf.reduce_mean(panel * p_coef)

    with tf.Session() as sess:
        tf.summary.FileWriter("logs/run_classifier/", sess.graph)
        logger.info("=======================save graph===========================")
        return loss, per_example_loss, logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps,
                     use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    # 返回函数自身, 这些参数都是写死的？？？？
    #     features,  # 输入的特征数据
    #     labels,  # 输入的标签数据
    #     mode,  # train、evaluate或predict
    #     params  # 超参数，对应上面Estimator传来的参数
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        # features ？？？
        """The `model_fn` for TPUEstimator."""
        logger.info("******************************* Features ***********************************")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))

        # 获取 输入的的对应参数
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        Q_mask = features["Q_mask"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 模型搭建
        logger.info(Q_mask)
        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings,
            Q_mask=Q_mask)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            # 加载pre train的模型
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)

            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold

            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logger.info("**** Trainable Variables ****")
        for var in tvars:
            # 将加载的模型的参数进行输出展示
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logger.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 训练
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(  # 黑科技？？？？？？？？？？
                mode=mode,
                loss=total_loss,
                train_op=train_op,  # ？？？？
                scaffold_fn=scaffold_fn
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                # 计算一个eval的准确率，以及一个平均loss
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,  ## ？？？
                scaffold_fn=scaffold_fn
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    all_Q_mask = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_Q_mask.append(feature.Q_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "Q_mask":
                tf.constant(
                    all_Q_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    # tf.logging.set_verbosity(logger.info)

    processors = {
        "zhejiang": ZhejiangProcesser,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # fine tuning 句子的长度不能比 pre train 长
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # 创建输出文件夹
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # 必须为每一task定义一个processor
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # 选取指定的数据
    logger.info("数据集合是: %s" % task_name)
    processor = processors[task_name]()

    # 事先定义好分类类别的label集合
    label_list = processor.get_labels()
    logger.info(label_list)

    # 加载词典，并将词典编号处理
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # 忽略tpu
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    tpu_config = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host
    )

    # RunConfig的设置
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tpu_config
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    # 获取训练的数据
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # 学习速率的确定？？？
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # 获取函数
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size
    )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
        logger.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder
        )

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
        logger.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder
        )

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            logger.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    # set paras data for test
    # FLAGS.task_name = "cola"
    # FLAGS.bert_config_file = "uncased_L-12_H-768_A-12/bert_config.json"
    # FLAGS.data_dir = "data/cola_public/format"
    # FLAGS.vocab_file = "uncased_L-12_H-768_A-12/vocab.txt"
    # FLAGS.output_dir = "tmp/cola"
    # FLAGS.init_checkpoint = "uncased_L-12_H-768_A-12/bert_model.ckpt"
    # FLAGS.do_train = True
    # FLAGS.do_eval = True
    # FLAGS.do_predict = True

    print_falgs(FLAGS)
    # logger.info(FLAGS.is_parsed())

    # logger.info("data dir: %s" % FLAGS.data_dir)
    # logger.info("bert_config_file dir: %s" % FLAGS.bert_config_file)
    # logger.info("task name: %s" % FLAGS.task_name)

    # 仅仅利用命令行运行上述的参数才会有效
    # main(None)  # 如果不在terminal中执行，默认参数没有用，上面也是

    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()  # 执行程序中的main函数并解析参数

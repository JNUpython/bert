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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
from mylogger import logger

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", "./sample_text.txt",
                    "pre train 的文本数据， 可以是多个用逗号分隔的文件， 每个文件用很多doc 用空行分开"
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("output_file", "./tmp/tf_examples.tfrecord",
                    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", "./tmp/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")

flags.DEFINE_bool("do_whole_word_mask", False,
                  "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128,
                     "Maximum sequence length.")

# 一个句子被mask掉的词汇上限
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345,
                     "Random seed for data generation.")
# 每个句子被重复利用的次数
flags.DEFINE_integer("dupe_factor", 10,
                     "Number of times to duplicate the input data (with different masks).")
# ？？？？max_predictions_per_seq
flags.DEFINE_float("masked_lm_prob", 0.15,
                   "Masked LM probability.")

# 虽然存在max_seq_length， 但是真实的句子并不是都能达到这个长度，生成一定长度读序列样本
flags.DEFINE_float("short_seq_prob", 0.1,
                   "Probability of creating sequences which are shorter than the maximum length.")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = "train instance:\n"
        s += "tokens: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files):
    """
    值得学习的一种将数据预处理结果保存的方法
    Create TF example files from `TrainingInstance`s."""
    writers = []
    # 可以将数据写出到多个文件
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        # 将文本转化为对应的id
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)

        # 被 >????  1 代表什么
        input_mask = [1] * len(input_ids)

        # 上下句子
        segment_ids = list(instance.segment_ids)

        assert len(input_ids) <= max_seq_length

        # padding 0
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)

        # 将mask掉的词汇转化为id
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        # 等权作用？？？？用于损失函数？？？
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        # task2：
        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        # 被mask的文本序列：id
        features["input_ids"] = create_int_feature(input_ids)
        # ？？？
        features["input_mask"] = create_int_feature(input_mask)
        # A B 两个句子对用id
        features["segment_ids"] = create_int_feature(segment_ids)
        # 句子中被mask掉的位置
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        # 被mask掉的位置对应id list
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        # 权重有何用
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        # 是不是下一句
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        # 序列化Example, 并写入到writer
        writers[writer_index].write(tf_example.SerializeToString())
        # next writer
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 3:
            # 对样本进行可视化展示
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    # 这样才能封装到tf.Example
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]  # 保存所有docments， 每个doc内容按行（每行是一个完整的句子）保存在一个list里面，并且每个句子被wordPeice，句子也是由list构成

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                # 读取每一行对一些特殊字符进行处理，
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                # logger.info(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents， 并且打乱文本的顺序，但是不打乱doc内部句子顺序
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    logger.info("a doc after token:")
    logger.info(all_documents[0])

    vocab_words = list(tokenizer.vocab.keys())
    # 样本例子
    instances = []
    for _ in range(dupe_factor):
        # 每个文本进行重复利用
        for document_index in range(len(all_documents)):
            instances.extend(
                # 用 doc 创造样本
                create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                               masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
            )

    rng.shuffle(instances)
    return instances


def create_instances_from_document(all_documents,  # 所有docs
                                   document_index,  # 用来docs中一个被利用创造train样本的doc
                                   max_seq_length,
                                   short_seq_prob,
                                   masked_lm_prob,
                                   max_predictions_per_seq,
                                   vocab_words,
                                   rng
                                   ):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3  # 真实句子最大

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        # 一定比例，生成一个随机更加短的数字， 这么做的作用还不是很明白？？？？为了更接近真实文本分布样本，
        # 并不是AB两部分长度均能达到maxlen
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    # logger.info(document)
    # logger.info(len(document))
    while i < len(document):
        segment = document[i]  # doc 的一个句子
        logger.info(i)
        # logger.info(segment)
        current_chunk.append(segment)  # A + B 候选样本
        current_length += len(segment)
        # 一个doc的最后一句, 句子长度加起来已经大于目标最大长度了然后将这些句子合成一个
        if i == len(document) - 1 or current_length >= target_seq_length:
            logger.info(segment)
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                # 一个样本多个句子组成，
                a_end = 1
                if len(current_chunk) >= 2:
                    # 随机生成一个数，将目前满足长度的样本句子，切分成 AB两部分
                    a_end = rng.randint(1, len(current_chunk) - 1)

                # 第一部分
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # 第二部分：用于第二个任务预测下一句
                tokens_b = []
                # Random next
                is_random_next = False
                # 5：5的比例生成一个随机的句子作为下一部分
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    # 整个doc循环完，仅仅可能构建一个样本，或者满足随机的概率
                    is_random_next = True

                    # 现在a已经 确定，获取b最大允许长度
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    random_document_index = None
                    for _ in range(10):
                        # 从库里选取一个随机doc
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    # 随机截取随机文本中的一些满足长度的b作为随机下一句
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    # 由于下一句是用随机生成的所current_chunk准备的B部分没有用到，这里并不浪费，将I往前调到没有用的地方
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                # 两者句子长度加起来可能大于最大长度
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []  # 真正的训练样本
                segment_ids = []  # 训练样本转化为id
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                # A 添加完毕
                tokens.append("[SEP]")
                segment_ids.append(0)  # A - 0

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)  # B - 1

                # 对样本进行mask
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                # 训练数据
                instance = TrainingInstance(
                    tokens=tokens,  # 训练输入
                    segment_ids=segment_ids,  # AB句子位置编号
                    is_random_next=is_random_next,  # 是否是真实下一个text（一个或者多个句子组成）
                    masked_lm_positions=masked_lm_positions,  # mask的index
                    masked_lm_labels=masked_lm_labels)  # mask掉的真实词汇
                instances.append(instance)
                # logger.info(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens,
                                 masked_lm_prob,
                                 max_predictions_per_seq,
                                 vocab_words,
                                 rng):
    """Creates the predictions for the masked LM objective.
    :returns： 真实句子， mask的index， mask之后的"""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        # 对词汇进行编号处理
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##"):
            # 由于采用wordPiece 一个词汇拆分成多个并用##开头，所以词汇用list保存
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    # 打乱编号 为了随机mask
    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []  # 被mask处理之后的tokens
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            # whole word = true 如果这个词汇被分成太多直接导致，超过预测长度，跳过这个词汇
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    # 将mask后的根据index还原到原来的句子顺序
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        # AB长度大于定的最大长度，找出最大长度的
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            # 对最大长度进行一定概率删除头部, A B 两者的长度加起来满足条件就不会再进行删除
            del trunc_tokens[0]
        else:
            # 为什么了进行清空??????
            trunc_tokens.pop()


def main(_):
    # 设置tf的logger日志级别

    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        # 获取匹配模式match的文件list
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)
        # logger.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)

    # 训练样本：  这些数据现在都在内存中？？？？
    instances = create_training_instances(
        input_files,
        tokenizer,
        FLAGS.max_seq_length,
        FLAGS.dupe_factor,
        FLAGS.short_seq_prob,
        FLAGS.masked_lm_prob,
        FLAGS.max_predictions_per_seq,
        rng
    )

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    # 训练样本的保存
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    # 必须要的参数，原本默认均为None，已经给出相应默认值
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()

"""
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
"""

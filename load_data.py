import os
import json

import tensorflow as tf
import numpy as np

from util import FLAGS


def get_vocab_dict(input_file=FLAGS.vocab_dict):
    # 返回 {word: id} 字典
    words_dict = {}
    with open(input_file) as f:
        for word in f:
            words_dict[word.strip()] = len(words_dict)
    return words_dict



def get_word_vector(input_file=FLAGS.word_vector):
    word_vectors = []
    with open(input_file) as f:
        for line in f:
            line = [float(v) for v in line.strip().split()]
            word_vectors.append(line)
    return word_vectors


def normalize(inputs, threshold_length):
    inputs_ = np.zeros(shape=[threshold_length], dtype=np.int64)  # == PAD

    for i, element in enumerate(inputs):
        if i >= threshold_length:
            break
        inputs_[i] = element
    return inputs_


# output batch_major data
def batch(inputs, threshold_length):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    '''
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
        max_sequence_length = min(max_sequence_length, threshold_length)
    '''
    max_sequence_length = threshold_length

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            if j >= threshold_length:
                sequence_lengths[i] = max_sequence_length
                break
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    # inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_batch_major


class LoadTrainData(object):
    def __init__(self, vocab_dict, data_path,
                 query_len_threshold, doc_len_threshold):
        self.vocab_dict = vocab_dict
        self.data_path = data_path
        self.doc_len_threshold = doc_len_threshold  # 句子长度限制
        self.query_len_threshold = query_len_threshold

        self.tfrecords = FLAGS.train_tfrecords

        self._data_to_tfrecords()

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res


    def _data_to_tfrecords(self):
        if os.path.exists(self.tfrecords):
            print("Exist tfrecords.")
            return
        print("start data to tfrecords for train ...")
        writer = tf.python_io.TFRecordWriter(self.tfrecords)
        with open(self.data_path) as f:
            for line in f:
                line = line.strip().split(',')
                ori_query = line[1].split()
                ori_pos_ans = line[3].split()
                ori_neg_ans = line[5].split()
                query = list(map(self._word_2_id, line[1].split()))
                pos_ans = list(map(self._word_2_id, line[3].split()))
                neg_ans = list(map(self._word_2_id, line[5].split()))

                docs = [ori_pos_ans, ori_neg_ans]
                features_local = np.array([])
                for doc in docs:
                    local_match = np.zeros(shape=[self.query_len_threshold, self.doc_len_threshold], dtype=np.int64)
                    for i in range(min(self.query_len_threshold, len(ori_query))):
                        for j in range(min(self.doc_len_threshold, len(doc))):
                            if ori_query[i] == doc[j]:
                                local_match[i, j] = 1
                    local_match = local_match.reshape([self.query_len_threshold * self.doc_len_threshold])
                    features_local = np.concatenate((features_local, local_match))
                query = normalize(query, self.query_len_threshold)
                pos_ans = normalize(pos_ans, self.doc_len_threshold)
                neg_ans = normalize(neg_ans, self.doc_len_threshold)
                features_local = features_local.astype(int)

                example = tf.train.Example(features=tf.train.Features(feature={
                    "query": tf.train.Feature(int64_list=tf.train.Int64List(value=query)),
                    "pos_ans": tf.train.Feature(int64_list=tf.train.Int64List(value=pos_ans)),
                    "neg_ans": tf.train.Feature(int64_list=tf.train.Int64List(value=neg_ans)),
                    "features_local": tf.train.Feature(int64_list=tf.train.Int64List(value=features_local))
                }))
                writer.write(example.SerializeToString())
        writer.close()

    def read_and_decode(self):
        print("read_and_decode...")
        filename_queue = tf.train.string_input_producer([self.tfrecords])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               "query": tf.FixedLenFeature([self.query_len_threshold], tf.int64),
                                               "pos_ans": tf.FixedLenFeature([self.doc_len_threshold], tf.int64),
                                               "neg_ans": tf.FixedLenFeature([self.doc_len_threshold], tf.int64),
                                               "features_local": tf.FixedLenFeature([2 * self.query_len_threshold *
                                                                                     self.doc_len_threshold], tf.int64)
                                           })
        query = features['query']
        pos_ans = features['pos_ans']
        neg_ans = features['neg_ans']
        features_local = tf.reshape(features['features_local'], [2, FLAGS.query_len_threshold, self.doc_len_threshold])


        docs = [pos_ans, neg_ans]

        return features_local, query, docs


class LoadTestData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, doc_len_threshold, batch_size):
        self.vocab_dict = vocab_dict
        self.data_path = data_path
        self.query_len_threshold = query_len_threshold
        self.doc_len_threshold = doc_len_threshold
        self.batch_index = 0
        self.data = open(data_path, 'r').readlines()
        self.data_size = len(self.data)
        self.batch_size = batch_size

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res

    def next_batch(self):
        if self.batch_size == -1:
            self.batch_size = 200
            self.data_size = self.batch_size * 5
        while (self.batch_index + 1) * self.batch_size <= self.data_size:
            batch_data = self.data[self.batch_index * self.batch_size: (self.batch_index + 1) * self.batch_size]
            self.batch_index += 1
            queries = []
            query_ids = []
            answers = []
            answers_ids = []
            answers_label = []
            batch_features_local = []

            for line in batch_data:
                line = json.loads(line)
                passages = line['passages']
                query_id = line['query_id']
                local_query = list(line['query'].split())
                query = list(map(self._word_2_id, line['query'].split()))
                for passage in passages:
                    passage_id = passage['passage_id']
                    label = passage['label']
                    local_passage = list(passage['passage_text'].split())
                    passage_text_list = list(map(self._word_2_id, passage['passage_text'].split()))
                    queries.append(query)
                    query_ids.append(query_id)
                    answers_ids.append(passage_id)
                    answers_label.append(label)
                    answers.append(passage_text_list)

                    local_match = np.zeros(shape=[self.query_len_threshold, self.doc_len_threshold], dtype=np.int32)
                    for i in range(min(self.query_len_threshold,len(local_query))):
                        for j in range(min(self.doc_len_threshold,len(local_passage))):
                            if local_query[i] == local_passage[j]:
                                local_match[i,j] = 1
                    batch_features_local.append(local_match)



            queries = batch(queries, self.query_len_threshold)
            answers = batch(answers, self.doc_len_threshold)
            yield batch_features_local, (query_ids, queries), (answers_ids, answers, answers_label)

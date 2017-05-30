import numpy as np
import json


def get_vocab_dict(input_file="../data/word_dict.txt"):
    # 返回 {word: id} 字典
    words_dict = {}
    num = 0
    with open(input_file) as f:
        for word in f:
            words_dict[word.strip()] = num
            num += 1

    return words_dict


def get_word_vector(input_file="../data/vectors_word.txt"):
    word_vectors = []
    with open(input_file) as f:
        for line in f:
            line = [float(v) for v in line.strip().split()]
            word_vectors.append(line)
    return word_vectors

# output batch_major data
def batch(inputs, threshold_length, max_sequence_length=None):
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
    #inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major


class LoadTrainData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, doc_len_threshold, batch_size=64):
        self.vocab_dict = vocab_dict
        self.data_path = data_path
        self.batch_size = batch_size
        self.doc_len_threshold = doc_len_threshold  # 句子长度限制
        self.query_len_threshold = query_len_threshold
        self.data = open(self.data_path, 'r').readlines()
        self.batch_index = 0
        print("len data: ", len(self.data))

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res

    def next_batch(self, shuffle=True):
        self.batch_index = 0
        data = np.array(self.data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1
        print("training_set:", data_size, num_batches_per_epoch)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        while self.batch_index < num_batches_per_epoch \
                and (self.batch_index + 1) * self.batch_size <= data_size:
            query_ids = []
            queries = []
            doc_ids = []
            docs = []
            pos_answers = []
            neg_answers = []
            batch_features_local = []
            start_index = self.batch_index * self.batch_size
            self.batch_index += 1
            end_index = min(self.batch_index * self.batch_size, data_size)
            batch_data = shuffled_data[start_index:end_index]

            for line in batch_data.tolist():
                line = line.split(',')
                query_id = int(line[0])
                query = list(map(self._word_2_id, line[1].split()))
                pos_id = int(line[2])
                pos_ans = list(map(self._word_2_id, line[3].split()))
                neg_id = int(line[4])
                neg_ans = list(map(self._word_2_id, line[5].split()))
                doc_id = [pos_id, neg_id]
                query_ids.append(query_id)
                queries.append(query)
                pos_answers.append(pos_ans)
                neg_answers.append(neg_ans)
                doc_ids.append(doc_id)

                features_local = []
                query = list(line[1].split())
                pos_doc = list(line[3].split())
                neg_doc = list(line[5].split())
                two_doc=[pos_doc,neg_doc]
                for doc in two_doc:
                    local_match = np.zeros(shape=[self.query_len_threshold, self.doc_len_threshold], dtype=np.int32)
                    for i in range(min(self.query_len_threshold,len(query))):
                        for j in range(min(self.doc_len_threshold,len(doc))):
                            if query[i]==doc[j]:
                                local_match[i,j] = 1
                    features_local.append(local_match)
                batch_features_local.append(features_local)

            queries = batch(queries, self.query_len_threshold)
            pos_answers = batch(pos_answers, self.doc_len_threshold)
            neg_answers = batch(neg_answers, self.doc_len_threshold)
            for (pos,neg) in zip(pos_answers, neg_answers):
                docs.append([pos, neg])

            yield batch_features_local, (query_ids, queries), (doc_ids, docs)


class LoadTestData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, doc_len_threshold, batch_size):
        self.vocab_dict = vocab_dict
        self.data_path = data_path
        self.query_len_threshold = query_len_threshold
        self.doc_len_threshold = doc_len_threshold
        self.index = 0
        self.data = open(data_path, 'r').readlines()
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.cnt = 0

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res

    def next_batch(self):
        if self.batch_size == -1:
            self.batch_size = 200
            self.data_size = self.batch_size*15
        while (self.index ) * self.batch_size < self.data_size:
            if (self.index + 1) * self.batch_size <= self.data_size:
                batch_data = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            else:
                batch_data = self.data[self.index * self.batch_size: self.data_size]
            self.index += 1
            queries = []
            query_ids = []
            answers = []
            answers_ids = []
            answers_label = []
            batch_features_local = []

            for line in batch_data:
                self.cnt += 1
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

        print("self.cnt:", self.cnt)

import os

import tensorflow as tf
import pandas as pd
import numpy as np

from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from tester import test
from util import FLAGS
from tester import dcg_k, normalized_dcg_k

# 加载词典
vocab_dict = get_vocab_dict()
word_vectors = get_word_vector()
vocab_size = len(vocab_dict)

"""
dev_set = LoadTestData(vocab_dict, "../data/dev.json",
                       query_len_threshold=FLAGS.query_len_threshold,
                       doc_len_threshold=FLAGS.doc_len_threshold,
                       batch_size= FLAGS.batch_size)
"""
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

Norm_DCG3 = []
Norm_DCG5 = []
Norm_FULL =[]
DCG_3 = []
DCG_5 = []
DCG_full = []

zero_3 = 0
zero_5 = 0

with tf.Session(config=config) as sess:
    """
    model = Model(max_query_word=FLAGS.query_len_threshold,
                  max_doc_word=FLAGS.doc_len_threshold,
                  num_docs=2,
                  word_vec_initializer=word_vectors,
                  batch_size=FLAGS.batch_size,
                  vocab_size=vocab_size,
                  embedding_size=FLAGS.embedding_dim,
                  learning_rate=FLAGS.learning_rate,
                  filter_size=FLAGS.filter_size,
                  keep_prob=FLAGS.keep_prob)
    """
    # 此处需要根据model名字改
    log_dir = "../logs/lr0.001_bz128_mg1_1496330529/"
    model_path = os.path.join(log_dir, "model.ckpt-20.meta")

    # 加载结构，模型参数和变量
    print("importing model...")
    saver = tf.train.import_meta_graph(model_path)

    saver.restore(sess, tf.train.latest_checkpoint(log_dir))
    sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()
    '''
    # 根据次数输出的变量名和操作名确定下边取值的名字
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print(v.name)

    for op in sess.graph.get_operations():
        print(op.name)
    '''
    query = graph.get_tensor_by_name("Test_Inputs/query:0")
    doc = graph.get_tensor_by_name("Test_Inputs/doc:0")
    feature_local = graph.get_tensor_by_name("Test_Inputs/feature_local:0")

    score = graph.get_tensor_by_name("Ensemble_model_2/score:0")

    dev_set = LoadTestData(vocab_dict, "../data/dev.json", query_len_threshold=FLAGS.query_len_threshold,
                           doc_len_threshold=FLAGS.doc_len_threshold, batch_size=FLAGS.batch_size)

    for batch_data in dev_set.next_batch():
        batch_features_local, (test_query_ids, test_queries), \
         (answers_ids, answers, answers_label) = batch_data

        fd = {query: test_queries,
              doc: answers,
              feature_local: batch_features_local}

        res = sess.run([score], fd)

        res = list(zip(test_query_ids, answers_ids, answers_label, res[0].tolist()))

        unique_query_ids = list(set(test_query_ids))
        df = pd.DataFrame(res, columns=['query_id', 'passage_id', 'label', 'score'])

        out_frames = []
        for query_id in unique_query_ids:
            passages = df[df['query_id'] == query_id]
            rank = range(1, passages.count()['label'] + 1)

            # result = passages.sort(['score'], ascending=False).reset_index(drop=True)
            real = passages.sort_values(by=['label'], ascending=False).reset_index(drop=True)
            real['rank'] = rank
            real.drop('score', axis=1, inplace=True)
            result = passages.sort_values(by=['score'], ascending=False).reset_index(drop=True)
            result['rank'] = rank
            # result.drop('score', axis=1, inplace=True)

            dcg_3 = dcg_k(result, 3)
            dcg_5 = dcg_k(result, 5)
            dcg_full = dcg_k(result, rank[-1])
            norm_dcg_3 = normalized_dcg_k(result, real, 3)
            norm_dcg_5 = normalized_dcg_k(result, real, 5)
            norm_dcg_full = normalized_dcg_k(result, real, rank[-1])
            result = passages.sort_values(by=['passage_id'], ascending=True).reset_index(drop=True)
            out_frames.append(result)

            if dcg_5 < 0.1:
                zero_3 += 1
                zero_5 += 1
            elif dcg_3 < 0.1:
                zero_3 += 1

            Norm_DCG3.append(norm_dcg_3)
            Norm_DCG5.append(norm_dcg_5)
            Norm_FULL.append(norm_dcg_full)

            DCG_3.append(dcg_3)
            DCG_5.append(dcg_5)
            DCG_full.append(dcg_full)

    dcg_3_mean = np.mean(DCG_3)
    dcg_5_mean = np.mean(DCG_5)
    dcg_full_mean = np.mean(DCG_full)

    norm_dcg_3_mean = np.mean(Norm_DCG3)
    norm_dcg_5_mean = np.mean(Norm_DCG5)
    norm_dcg_full_mean = np.mean(Norm_FULL)

    print(len(DCG_3))
    print(len(DCG_5))

    print("number of Zero DCG@3: ", zero_3)
    print("number of Zero DCG@5: ", zero_5)
    print("DCG@3 Mean: ", dcg_3_mean, "\tNorm: ", norm_dcg_3_mean)
    print("DCG@5 Mean: ", dcg_5_mean, "\tNorm: ", norm_dcg_5_mean)
    print("DCG@full Mean: ", dcg_full_mean, "\tNorm: ", norm_dcg_full_mean)
    print("=" * 60)


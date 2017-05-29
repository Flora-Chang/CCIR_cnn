import io
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from util import FLAGS

from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from tester import test

# 加载词典
vocab_dict = get_vocab_dict()
word_vectors = get_word_vector()
vocab_size = len(vocab_dict)
#print("vocab_size: ",vocab_size)
#print("word_vector: ", len(word_vectors))
train_set = LoadTestData(vocab_dict, "../data/train.json", query_len_threshold=FLAGS.query_len_threshold,\
                         doc_len_threshold=FLAGS.doc_len_threshold, batch_size= FLAGS.batch_size)
dev_set = LoadTestData(vocab_dict, "../data/dev.json", query_len_threshold=FLAGS.query_len_threshold,\
                       doc_len_threshold=FLAGS.doc_len_threshold, batch_size= FLAGS.batch_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=config) as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ",  time.asctime(time.localtime(time.time()) ))
    model_name = "lr{}_bz{}_mg{}_{}".format(FLAGS.learning_rate,
                                                             FLAGS.batch_size,
                                                             FLAGS.margin,
                                                             timestamp)

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

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    checkpoint_dir = './save_6000/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else :
        print("save file not exits")
        pass

    dev_set = LoadTestData(vocab_dict, "../data/dev_cnn.json", query_len_threshold=FLAGS.query_len_threshold, \
                           doc_len_threshold=FLAGS.doc_len_threshold, batch_size=FLAGS.batch_size)

    _, _, _ = test(sess, model, dev_set, filename="../data/result/dev_result.csv")
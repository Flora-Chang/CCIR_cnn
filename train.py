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

training_set = LoadTrainData(vocab_dict,
                          data_path=FLAGS.training_set,
                          query_len_threshold=FLAGS.query_len_threshold,
                          doc_len_threshold=FLAGS.doc_len_threshold)

features_local, query, docs = training_set.read_and_decode()
features_local_batch, query_batch, docs_batch = \
    tf.train.shuffle_batch([features_local, query, docs],
                           batch_size=FLAGS.batch_size,
                           capacity=20000,
                           min_after_dequeue=2000)

dev_set = LoadTestData(vocab_dict, "../data/dev.json",
                       query_len_threshold=FLAGS.query_len_threshold,
                       doc_len_threshold=FLAGS.doc_len_threshold,
                       batch_size= FLAGS.batch_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config) as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ", timestamp)
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

    log_dir = "../logs/" + model_name
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    steps = []
    train_DCG_3 = []
    train_DCG_5 = []
    train_DCG_full = []
    val_DCG_3 = []
    val_DCG_5 = []
    val_DCG_full = []
    test_DCG_3 = []
    test_DCG_5 = []
    test_DCG_full = []

    step = 0
    total_steps = FLAGS.total_training_num // FLAGS.batch_size
    print("total steps number: ", total_steps)
    num_epochs = FLAGS.num_epochs
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        for i in range(total_steps):
            features_local, queries, docs = sess.run([features_local_batch, query_batch, docs_batch])
            labels = np.zeros(shape=[FLAGS.batch_size, 2], dtype=np.float32)
            for label in labels:
                label[0] = 1

            feed_dict = {model.features_local: features_local,
                         model.queries: queries,
                         model.docs: docs,
                         model.labels: labels}
            #_, loss, summary = sess.run([model.train_op, model.loss, model.merged_summary_op], feed_dict)
            _, loss, score_pos, score_neg, subs, summary =\
                sess.run([model.optimize_op, model.loss, model.score_pos,
                          model.score_neg, model.sub, model.merged_summary_op],
                         feed_dict)

            train_writer.add_summary(summary, step)

            if step % 300 == 0:
                print(step, " - loss:", loss)

                train_set = LoadTestData(vocab_dict, "../data/0.1_train.json",
                                         query_len_threshold=FLAGS.query_len_threshold,
                                         doc_len_threshold=FLAGS.doc_len_threshold, batch_size=-1)

                print("On training set:\n")
                dcg_3, dcg_5, dcg_full = test(sess, model, train_set, filename="train_result.csv")
                train_DCG_3.append(dcg_3)
                train_DCG_3.append(dcg_5)
                train_DCG_full.append(dcg_full)


                dev_set = LoadTestData(vocab_dict, "../data/dev.json", query_len_threshold=FLAGS.query_len_threshold,
                                       doc_len_threshold=FLAGS.doc_len_threshold, batch_size=FLAGS.batch_size)
                print("On validation set:\n")

                dcg_3, dcg_5, dcg_full = test(sess, model, dev_set, filename="dev_result.csv")
                val_DCG_3.append(dcg_3)
                val_DCG_5.append(dcg_5)
                val_DCG_full.append(dcg_full)

            step += 1


        '''
        print("On test set:\n")
        dcg_3, dcg_5, dcg_full = test(sess, model, test_set)

        test_DCG_3.append(dcg_3)
        test_DCG_5.append(dcg_5)
        test_DCG_full.append(dcg_full)
        '''

        '''
        saver = tf.train.Saver(tf.global_variables())

        saver_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"), step)
        '''
    coord.request_stop()
    coord.join(threads)

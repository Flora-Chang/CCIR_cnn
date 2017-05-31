import tensorflow as tf


flags = tf.app.flags

# Model parameters

flags.DEFINE_integer("filter_size", 64, "the num of filters of CNN")
flags.DEFINE_integer("embedding_dim", 100, "words embedding size")
flags.DEFINE_float("keep_prob", 0.8, "dropout keep prob")

# Training / test parameters
flags.DEFINE_integer("total_training_num", 449073, "total number of training set")
flags.DEFINE_integer("query_len_threshold", 20, "threshold value of query length")
flags.DEFINE_integer("doc_len_threshold", 200, "threshold value of document length")
flags.DEFINE_integer("batch_size", 128, "batch size")

flags.DEFINE_integer("num_epochs", 3, "number of epochs")

flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_float("margin", 1, "cos margin")

flags.DEFINE_string("vocab_dict", "../data/word_dict.txt", "vocab dict path")
flags.DEFINE_string("word_vector", "../data/vectors_word.txt", "word vectors file path")
flags.DEFINE_string("train_tfrecords", "../data/tfrecords/train.cnn.tfrecords", "tfrecords of training set path")
flags.DEFINE_string("dev_tfrecords", "../data/tfrecords/dev.cnn.tfrecords", "tfrecords of dev set path")

flags.DEFINE_string("training_set", "../data/train.csv", "training set path")
flags.DEFINE_string("dev_set", "../data/dev.json", "dev set path")

FLAGS = flags.FLAGS
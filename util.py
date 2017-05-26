import tensorflow as tf


flags = tf.app.flags

# Model parameters
flags.DEFINE_integer("filter_size", 128, "the num of filters of CNN")
flags.DEFINE_integer("embedding_dim", 100, "words embedding size")
flags.DEFINE_float("keep_prob", 0.8, "dropout keep prob")



# Training / test parameters
flags.DEFINE_integer("query_len_threshold", 15, "threshold value of query length")
flags.DEFINE_integer("doc_len_threshold", 200, "threshold value of document length")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_epochs", 8, "number of epochs")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_float("margin", 1, "cos margin")
flags.DEFINE_string("training_set", "../data/train.csv", "training set path")

FLAGS = flags.FLAGS
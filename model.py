import tensorflow as tf


class Model(object):
    def __init__(self, max_query_word, max_doc_word, num_docs, word_vec_initializer, batch_size, filter_size,
                 vocab_size, embedding_size, learning_rate, keep_prob):
        self.word_vec_initializer = word_vec_initializer
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.max_query_word = max_query_word
        self.max_doc_word = max_doc_word
        self.num_docs = num_docs
        self.filter_size = filter_size
        self.local_output = None
        self.distrib_output = None

        self._input_layer()
        self.optimizer(self.features_local, self.queries, self.docs)
        self.test(self.feature_local, self.query, self.doc)
        #self.merged_summary_op = tf.summary.merge([self.sm_loss_op, self.sm_emx_op])
        self.merged_summary_op = tf.summary.merge([self.sm_loss_op])

    def _input_layer(self):
        #with tf.variable_scope('Inputs'):
        with tf.variable_scope('Train_Inputs'):
            self.features_local = tf.placeholder(dtype=tf.float32,
                                                 shape=(None, self.num_docs, self.max_query_word, self.max_doc_word),
                                                 name='features_local')
            self.queries = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='queries')
            self.docs = tf.placeholder(dtype=tf.int32, shape=(None, self.num_docs, self.max_doc_word), name='docs')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_docs], name='labels')

        with tf.variable_scope('Test_Inputs'):
            self.feature_local = tf.placeholder(dtype=tf.float32,
                                                shape=(None, self.max_query_word, self.max_doc_word),
                                                name='feature_local')
            self.query = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='query')
            self.doc = tf.placeholder(dtype=tf.int32, shape=(None, self.max_doc_word), name='doc')

    def _embed_layer(self, query, doc):
        with tf.variable_scope('Embedding_layer'), tf.device("/cpu:0"):
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                                    initializer=self.word_vec_initializer,
                                                    dtype=tf.float32,
                                                    trainable=False)
            self.sm_emx_op = tf.summary.histogram('EmbeddingMatrix', self.embedding_matrix)
            embedding_query = tf.nn.embedding_lookup(self.embedding_matrix, query)
            embedding_doc = tf.nn.embedding_lookup(self.embedding_matrix, doc)
            return embedding_query, embedding_doc

    def local_model(self, features_local, is_training=True, reuse=False):
        with tf.variable_scope('local_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            features_local = tf.reshape(features_local, [-1, self.max_query_word, self.max_doc_word]) #[?,15,200]
            conv = tf.layers.conv1d(inputs=features_local, filters=self.filter_size, kernel_size=[ 1], \
                                    activation=tf.nn.tanh) #[?,15,1,self.filter_size]
            conv = tf.reshape(conv, [-1,self.filter_size*self.max_query_word]) #[?,15*self.filter_size]
            dense1 = tf.layers.dense(inputs=conv, units=self.filter_size, activation=tf.nn.tanh) #[?, self.filter_size]
            #dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training) #extra add
            dense2 = tf.layers.dense(inputs=dense1, units=self.filter_size, activation=tf.nn.tanh)
            #dropout = tf.layers.dropout(inputs=dense2, rate=self.keep_prob, training=is_training)
            #dense3 = tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.tanh)  #[?,1]
            #self.local_output =  dense3
            self.local_output = dense2
            return self.local_output


    def distrib_model(self, query, doc, is_training=True, reuse=False):
        with tf.variable_scope('Distrib_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            embedding_query, embedding_doc = self._embed_layer(query=query, doc=doc)
            print("embeded_query: ", embedding_query)
            print("embeded_doc: ", embedding_doc)

            with tf.variable_scope('distrib_query'):
                print("in dis query")
                query = tf.reshape(embedding_query,
                                   [-1, self.max_query_word, self.embedding_size, 1])  # [?, max_query_word(=15), self.embedding_size,1]
                print("query: ", query)
                conv1 = tf.layers.conv2d(inputs=query, filters=self.filter_size,
                                         kernel_size=[3, self.embedding_size],
                                         activation=tf.nn.tanh, name="conv_query")  # [?,15-3+1,1, self.filter_size]
                print("conv1: ", conv1)
                pooling_size = self.max_query_word - 3 + 1
                print("pooling_size: ", pooling_size)
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_query")  # [?, 1,1 self.filter_size]
                print("pool1: ", pool1)
                pool1 = tf.reshape(pool1, [-1, self.filter_size])  # [?, self.filter_size]
                print("pool1: ", pool1)
                dense1 = tf.layers.dense(inputs=pool1, units=self.filter_size, activation=tf.nn.tanh, name="fc_query")
                distrib_query = dense1  # [?, self.filter_size]

            with tf.variable_scope('distrib_doc'):
                print("in dis doc")
                doc = tf.reshape(embedding_doc, [-1, self.max_doc_word, self.embedding_size])
                print("doc: ", doc)
                conv1 = tf.layers.conv1d(inputs=doc,
                                         filters=self.filter_size,
                                         kernel_size=[3],
                                         activation=tf.nn.tanh,
                                         name="conv_doc1")  # [?, self.max_doc_word -3 +1, self.filter_size]
                print("conv1: ", conv1)
                pooling_size = 80
                pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[pooling_size], strides=[1])
                #[?, self.max_doc_word-3+1-pooling_size+1, self.filter_size]
                conv2 = tf.layers.conv1d(inputs=pool1, filters=self.filter_size, kernel_size=[1])
                self.distrib_doc = conv2 #like before
                #self.dims1 = self.max_doc_word - pooling_size -1
                #self.dims2 = (self.max_doc_word - pooling_size -1)*self.filter_size
                self.dims1 = self.max_doc_word - pooling_size -1
                self.dims2 = (self.max_doc_word - pooling_size -1 )*self.filter_size

            self.distrib_query = tf.tile(tf.expand_dims(self.distrib_query, 1), [1,self.dims1, 1])
            distrib = tf.multiply(self.distrib_query, self.distrib_doc) #[?, self.dims1, self.filter_size]
            distrib = tf.reshape(distrib,[-1,self.dims2]) #[?, self.dims2]
            fuly1 = tf.layers.dense(inputs=distrib, units=self.filter_size, activation=tf.nn.tanh)
            #drop = tf.layers.dropout(inputs=fuly1, rate=self.keep_prob, training=is_training)  # extra add
            fuly2 = tf.layers.dense(inputs=fuly1, units=self.filter_size, activation=tf.nn.tanh)
            #drop2 = tf.layers.dropout(inputs=fuly2, rate=self.keep_prob, training=is_training)
            #fuly3 = tf.layers.dense(inputs=drop2, units=1, activation=tf.nn.tanh)
            #self.distrib_output = fuly3 #[?, 1]
            self.distrib_output = fuly2  # [?, 1]
            print("distrib_output:",self.distrib_output)
            return self.distrib_output


    def ensemble_model(self, features_local, query, doc, is_training=True, reuse=False):
        with tf.variable_scope('emsemble_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            #self.model_output = tf.add(self.local_model(is_training=is_training, features_local = features_local,\
                                                        #reuse=reuse),self.distrib_model(is_training=is_training, \
                                                        #query=query,doc=doc,reuse=reuse))
            self.model_output = tf.concat([self.local_model(is_training=is_training, features_local=features_local, \
                                                            reuse=reuse),self.distrib_model(is_training=is_training, \
                                                            query=query,doc=doc,reuse=reuse)], axis=-1)
            fuly = tf.layers.dense(inputs=self.model_output, units=self.filter_size, activation=tf.nn.tanh)
            fuly1 = tf.layers.dense(inputs=fuly, units=1, activation=tf.nn.tanh)
            #self.model_output =  self.distrib_model(is_training=is_training, query=query, doc=doc,reuse=reuse)
            #self.model_output = self.local_model(is_training=is_training, features_local = features_local,reuse=reuse)

        #output = tf.nn.sigmoid(self.model_output)
        #output = self.model_output
        output = fuly1
        return output

    def optimizer(self, features_local, queries, docs):

        docs = tf.transpose(docs, [1, 0, 2], name="docs_transpose")  # [2, batch_size, words_num]
        features_local = tf.transpose(features_local, [1, 0, 2, 3], name="local_features_transpose")  # [2, batch_size, query_length, doc_length]

        self.score_pos = self.ensemble_model(features_local=features_local[0], query=queries,
                                             doc=docs[0], is_training=True, reuse=False)  # [batch_size, 1]
        self.score_neg = self.ensemble_model(features_local=features_local[1], query=queries,
                                             doc=docs[1], is_training=True, reuse=True)  # [batch_size, 1]
        #with tf.variable_scope("Optimizer"):
        with tf.name_scope("loss"):
            self.score_pos = tf.squeeze(self.score_pos, -1, name="squeeze_pos")  # [batch_size]
            self.score_neg = tf.squeeze(self.score_neg, -1, name="squeeze_neg")  # [batch_size]
            self.sub = tf.subtract(self.score_pos, self.score_neg, name="pos_sub_neg")

            self.losses = tf.maximum(0.0, tf.subtract(1.0, tf.subtract(self.score_pos, self.score_neg)))
            self.loss = tf.reduce_mean(self.losses)
            self.sm_loss_op = tf.summary.scalar('Loss', self.loss)

        with tf.name_scope("optimizer"):
            #self.optimize_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.optimize_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def test(self, feature_local, query, doc):
        self.score = self.ensemble_model(features_local=feature_local, query=query,
                                         doc=doc, is_training=False, reuse=True)
        print("score:", self.score)


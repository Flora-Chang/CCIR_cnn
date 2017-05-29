import tensorflow as tf


class Model(object):
    def __init__(self, max_query_word, max_doc_word,num_docs, word_vec_initializer, batch_size, filter_size, \
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
        #self.regularizer = tf.contrib.layers.l2_regularizer(0.001)
        self._input_layer()
        self.train(self.features_local, self.queries, self.docs)
        self.test(self.feature_local, self.query, self.doc)

    def _input_layer(self):
        with tf.variable_scope('Inputs'):
            self.features_local = tf.placeholder(dtype=tf.float32, shape=(None, self.num_docs,self.max_query_word, self.max_doc_word),name='features_local')
            self.queries = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='queries')
            self.docs = tf.placeholder(dtype=tf.int32, shape=(None, self.num_docs, self.max_doc_word), name='docs')
            self.doc = tf.placeholder(dtype=tf.int32,shape=(None,self.max_doc_word), name='doc')
            self.query = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='query')
            self.feature_local = tf.placeholder(dtype=tf.float32, shape=(None,self.max_query_word, self.max_doc_word), name='feature_local')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_docs],name='labels')
            print("input:",self.features_local, self.query)

    def _embed_layer(self, query, doc):
        with tf.variable_scope('Embedding_layer'):
            self.embedding_matrix = tf.get_variable(name='embedding_matrix', \
                                                    initializer=self.word_vec_initializer, \
                                                    dtype=tf.float32, \
                                                    trainable=False)
            embedding_query = tf.nn.embedding_lookup(self.embedding_matrix, query)
            embedding_doc = tf.nn.embedding_lookup(self.embedding_matrix, doc)
            return embedding_query, embedding_doc

    def local_model(self, features_local, is_training=True, reuse=False):
        with tf.variable_scope('local_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            features_local = tf.reshape(features_local, [-1, self.max_query_word, self.max_doc_word]) #[?,15,200]
            conv = tf.layers.conv1d(inputs=features_local, filters=self.filter_size, kernel_size=[ 1], \
                                    activation=tf.nn.relu) #[?,15,1,self.filter_size]
            conv = tf.reshape(conv, [-1,self.filter_size*self.max_query_word]) #[?,15*self.filter_size]
            dense1 = tf.layers.dense(inputs=conv, units=self.filter_size, activation=tf.nn.tanh) #[?, self.filter_size]
            dense2 = tf.layers.dense(inputs=dense1, units=self.filter_size, activation=tf.nn.tanh)
            dropout = tf.layers.dropout(inputs=dense2, rate=self.keep_prob, training=is_training)
            dense3 = tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.tanh)  #[?,1]
            self.local_output = dense3
            return self.local_output       
    '''

    def local_model(self, features_local, is_training=True, reuse=False):
        with tf.variable_scope('local_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            features_local = tf.reshape(features_local, [-1, self.max_query_word, self.max_doc_word, 1])  # [?,15,200]
            conv = tf.layers.conv2d(inputs=features_local, filters=self.filter_size, kernel_size=[3, 3], \
                                padding="same", activation=tf.nn.tanh)  # [?,15,1,self.filter_size]
            pool = tf.layers.max_pooling2d(conv, pool_size=[3,3], strides=[1,1])
            print("pool:", pool)
            pool = tf.reshape(pool, [-1, self.filter_size * (self.max_query_word-2)*(self.max_doc_word-2)])  # [?,15*self.filter_size]
            dense1 = tf.layers.dense(inputs=pool, units=self.filter_size*(self.max_query_word-1),
                                     activation=tf.nn.tanh)  # [?, self.filter_size]
            dense2 = tf.layers.dense(inputs=dense1, units=self.filter_size, activation=tf.nn.tanh)
            dropout = tf.layers.dropout(inputs=dense2, rate=self.keep_prob, training=is_training)
            dense3 = tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.tanh)  # [?,1]
            self.local_output = dense3
            return self.local_output
    '''

    def distrib_model(self, query, doc, is_training=True, reuse=False):
        with tf.variable_scope('distrib_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            embedding_query, embedding_doc = self._embed_layer(query=query, doc=doc)
            with tf.variable_scope('distrib_query'):
                query = tf.reshape(embedding_query, [-1, self.max_query_word, self.embedding_size, 1]) #[?, 15, self.embedding_size,1]
                conv1 = tf.layers.conv2d(inputs=query, filters=self.filter_size, \
                                         kernel_size=[3, self.embedding_size], activation=tf.nn.tanh) #[?,15-3+1,1, self.filter_size]
                #pooling_size = self.max_query_word - 3 + 1
                pooling_size = self.max_query_word - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[pooling_size, 1],strides=[1,1])  #[?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1,self.filter_size]) #[?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.filter_size, activation=tf.nn.tanh)
                self.distrib_query = dense1 #[?, self.filter_size]

            with tf.variable_scope('distrib_doc'):
                doc = tf.reshape(embedding_doc, [-1, self.max_doc_word, self.embedding_size])
                conv1 = tf.layers.conv1d(inputs=doc, filters=self.filter_size, \
                                         kernel_size=[3], activation=tf.nn.tanh) #[?, self.max_doc_word -3 +1, self.filter_size]
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
            #fuly2 = tf.layers.dense(inputs=fuly1, units=self.filter_size, activation=tf.nn.tanh)
            drop2 = tf.layers.dropout(inputs=fuly1, rate=self.keep_prob, training=is_training)
            fuly3 = tf.layers.dense(inputs=drop2, units=1, activation=tf.nn.tanh)
            self.distrib_output = fuly3 #[?, 1]
            print("distrib_output:",self.distrib_output)
            return self.distrib_output

    def ensemble_model(self, features_local, query, doc, is_training=True, reuse=False):
        with tf.variable_scope('emsemble_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self.model_output = tf.add(self.local_model(is_training=is_training, features_local = features_local,\
                                                        reuse=reuse),self.distrib_model(is_training=is_training, \
                                                        query=query,doc=doc,reuse=reuse))
            #self.model_output =  self.distrib_model(is_training=is_training, query=query, doc=doc,reuse=reuse)
            #self.model_output = self.local_model(is_training=is_training, features_local = features_local,reuse=reuse)
        #output = tf.nn.sigmoid(self.model_output)
        output = self.model_output
        return output

    def train(self, features_local, query, docs):
        docs_shape = tf.shape(docs)
        features_local_shape = tf.shape(features_local)
        print("docs_shape:",docs_shape, 'features_local_shape:',features_local_shape)
        docs = tf.transpose(docs, [1, 0, 2])
        features_local = tf.transpose(features_local, [1,0,2,3])
        print("features_local:",features_local)
        print("features_local0:", features_local[0])
        self.features0 = features_local[0]
        self.features1 = features_local[1]
        self.score1 = self.ensemble_model(features_local=features_local[0], query=query,\
                                          doc=docs[0],is_training=True,reuse=False)
        self.score2 = self.ensemble_model(features_local=features_local[1],query=query,\
                                          doc=docs[1],is_training=True,reuse=True)
        self.score1 = tf.squeeze(self.score1, -1)
        self.score2 = tf.squeeze(self.score2, -1)
        print("score1:",self.score1)
        print("score2:", self.score2)
        #self.result = tf.concat([self.score1, self.score2], axis=1)
        #self.max_score = tf.reduce_max(self.result)
        #print("self.result:", self.result)
        #print("self.labels:",self.labels)
        #self.loss = tf.reduce_mean(tf.square((self.result - self.labels)))
        self.sub = tf.subtract(self.score1, self.score2)
        #zero = tf.constant(0, shape=[self.batch_size], dtype=tf.float32)
        #margin = tf.constant(2.0, shape=[self.batch_size], dtype=tf.float32)
        #self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.score1, self.score2)))
        self.losses = tf.maximum(0.0, tf.subtract(1.0, tf.subtract(self.score1, self.score2)))
        #print("losses:",self.losses)
        #vars = tf.trainable_variables()
        #loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 0.01
        self.loss = tf.reduce_mean(self.losses)
        #self.loss = tf.reduce_mean(self.losses) + tf.contrib.layers.apply_regularization(self.regularizer)
        print("weights:",tf.GraphKeys.WEIGHTS)
        #self.loss = tf.reduce_sum(self.losses)
        #self.loss = tf.reduce_mean(tf.subtract(0.0,self.sub))
        print("loss:",self.loss)
        #self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def test(self, feature_local, query, doc):
        self.score = self.ensemble_model(features_local=feature_local,query=query,\
                                         doc=doc, is_training=False,reuse=True)
        self.score = tf.squeeze(self.score, -1)
        print("score:",self.score)

